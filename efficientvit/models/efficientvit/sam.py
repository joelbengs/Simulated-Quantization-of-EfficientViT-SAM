# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientvit.models.ptq.observer.base import BaseObserver
from segment_anything import SamAutomaticMaskGenerator
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.utils.amg import build_all_layer_point_grids
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.transforms.functional import resize, to_pil_image
import matplotlib.pyplot as plt

from efficientvit.models.efficientvit.backbone import EfficientViTBackbone, EfficientViTLargeBackbone
from efficientvit.models.nn import (
    ConvLayer,
    DAGBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResBlock,
    ResidualBlock,
    UpSampleLayer,
    build_norm,
    ## quantized basic layers ##
    QConvLayer,
    QConvLayerV2,
    ## quantized basic blocks ##
    QDSConv,
    QMBConv,
    QFusedMBConv,
    QResBlock,
    QLiteMLA,
    QEfficientViTBlock,
    ## there are no quantized functional blocks ##
)
from efficientvit.models.utils import build_kwargs_from_config, get_device

__all__ = [
    "SamPad",
    "SamResize",
    "SamNeck",
    "EfficientViTSamImageEncoder",
    "EfficientViTSam",
    "EfficientViTSamPredictor",
    "EfficientViTSamAutomaticMaskGenerator",
    "efficientvit_sam_l0",
    "efficientvit_sam_l1",
    "efficientvit_sam_l2",
    "efficientvit_sam_xl0",
    "efficientvit_sam_xl1",
    # quantized versions #
    "efficientvit_sam_l0_quant",
    "efficientvit_sam_l1_quant",
    "efficientvit_sam_l2_quant",
    "efficientvit_sam_xl0_quant",
    "efficientvit_sam_xl1_quant",
]


class SamPad:
    def __init__(self, size: int, fill: float = 0, pad_mode="corner") -> None:
        self.size = size
        self.fill = fill
        self.pad_mode = pad_mode

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        th, tw = self.size, self.size
        assert th >= h and tw >= w
        if self.pad_mode == "corner":
            image = F.pad(image, (0, tw - w, 0, th - h), value=self.fill)
        else:
            raise NotImplementedError
        return image

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size},mode={self.pad_mode},fill={self.fill})"


class SamResize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.size)
        return np.array(resize(to_pil_image(image), target_size))

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


class SamNeck(DAGBlock):
    def __init__(
        self,
        fid_list: list[str],
        in_channel_list: list[int],
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        middle_op: str,
        out_dim: int = 256,
        norm="bn2d",
        act_func="gelu",
    ):
        inputs = {}
        for fid, in_channel in zip(fid_list, in_channel_list):
            inputs[fid] = OpSequential(
                [
                    ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None),
                    UpSampleLayer(size=(64, 64)),
                ]
            )

        middle = []
        for _ in range(head_depth):
            if middle_op == "mb":
                block = MBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, act_func, None),
                )
            elif middle_op == "fmb":
                block = FusedMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                )
            elif middle_op == "res":
                block = ResBlock(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                )
            else:
                raise NotImplementedError
            middle.append(ResidualBlock(block, IdentityLayer()))
        middle = OpSequential(middle)

        outputs = {
            "sam_encoder": OpSequential(
                [
                    ConvLayer(
                        head_width,
                        out_dim,
                        1,
                        use_bias=True,
                        norm=None,
                        act_func=None,
                    ),
                ]
            )
        }

        super(SamNeck, self).__init__(inputs, "add", None, middle=middle, outputs=outputs)


class QSamNeck(DAGBlock):
    def __init__(
        self,
        fid_list: list[str],
        in_channel_list: list[int],
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        middle_op: str,
        out_dim: int = 256,
        norm="bn2d",
        act_func="gelu",
        config=None,
    ):
        inputs = {}
        for i, (fid, in_channel) in enumerate(zip(fid_list, in_channel_list), start=0):
            inputs[fid] = OpSequential(
                [
                    QConvLayer(
                        in_channel, 
                        head_width, 
                        kernel_size=1, 
                        norm=norm, 
                        act_func=None,
                        # configs
                        config = config,
                        stage_id='neck',
                        block_position = i,
                        layer_position = 0,
                        block_name="independent",
                        block_is_bottleneck=True, #no residual connection exists
                        block_is_neck=True,
                        ),
                    UpSampleLayer(size=(64, 64)),
                ]
            )

        middle = []
        for i in range(head_depth):
            if middle_op == "mb": # not used by any model
                block = QMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, act_func, None),
                    # configs
                    config = config,
                    stage_id='neck',
                    block_position=i+1+len(fid_list),
                    block_name=middle_op,
                    block_is_neck=True,
                )
            elif middle_op == "fmb": # used by all models
                block = QFusedMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                    # configs
                    config = config,
                    stage_id='neck',
                    block_position=i+len(fid_list),
                    block_name=middle_op,
                    block_is_neck=True,
                )
            elif middle_op == "res": # not used by any model
                block = QResBlock(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                    # configs
                    config = config,
                    stage_id='neck',
                    block_position=i+1+len(fid_list),
                    block_name=middle_op,
                    block_is_neck=True,
                )
            else:
                raise NotImplementedError
            middle.append(ResidualBlock(block, IdentityLayer()))
        middle = OpSequential(middle)

        outputs = {
            "sam_encoder": OpSequential(
                [
                    QConvLayer(
                        head_width,
                        out_dim,
                        1,
                        use_bias=True,
                        norm=None,
                        act_func=None,
                        # configs
                        config = config,
                        stage_id='neck',
                        block_position=len(fid_list)+head_depth,
                        layer_position=0,
                        block_name='independent',
                        block_is_bottleneck=True, #no residual connection exists
                        block_is_neck=True,
                    ),
                ]
            )
        }

        # Creates a DAGblock with the above specified operations
        super(QSamNeck, self).__init__(inputs, "add", None, middle=middle, outputs=outputs)


class EfficientViTSamImageEncoder(nn.Module):
    def __init__(self, backbone: EfficientViTBackbone or EfficientViTLargeBackbone, neck: SamNeck):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

        self.norm = build_norm("ln2d", 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feed_dict = self.backbone(x)
        feed_dict = self.neck(feed_dict)

        output = feed_dict["sam_encoder"]
        output = self.norm(output)
        return output


    '''
    # quantizes only specificed parts. Can be specified by stages, by block names, by the intersection of both. Can be specified to save bottlenecks
    # toggle the specified attribute for all modules in the intersection of stages and blocknames.
    # Exclude the 
    def toggle_selective_attribute(
            self, 
            attribute: str,
            attribute_goal_state=True,
            printout=False,
            stages=["unknown", "stage0", "stage1", "stage2", "stage4", "stage5", "neck"],
            block_position= [0,1,2,3,4,5,6,7,8,9],
            layer_position= [0,1,2,3,4,5,6,7,8,9],
            block_names=['independent', "res", "mb", "fmb", "att", "att@3", "att@5"], # could be more scales, must build a general solution for any scale
            spare_bottlenecks=False,
            spare_attention_qkv=False,
            spare_attention_scaling=False,
            spare_attention_projection=False,
            ):
        count_all = count_candidates = count_affected = 0
        for m in self.modules():
            count_all = count_all + 1
            if type(m) in [QConvLayer, QConvLayerV2]:
                count_candidates = count_candidates + 1
                # TODO: Make these set the attribute to False to protect human logic error
                if m.block_is_bottleneck and spare_bottlenecks:
                    if printout: print(f"spared bottleneck: {m.block_name} in {m.stage_id}", f"                module {count_all} of type {type(m)}")
                    continue # skips to the next iteration
                if m.block_name.startswith("att"):
                    if spare_attention_qkv and m.conv_is_attention_qkv:
                        if printout: print(f"spared QKV of {m.block_name} in {m.stage_id}", f"                module {count_all} of type {type(m)}")
                        continue
                    if spare_attention_scaling and m.conv_is_attention_scaling:
                        if printout: print(f"spared scaling of {m.block_name} in {m.stage_id}", f"                module {count_all} of type {type(m)}")
                        continue
                    if  spare_attention_projection and m.conv_is_attention_projection:
                        if printout: print(f"spared projection of {m.block_name} in {m.stage_id}", f"                module {count_all} of type {type(m)}")
                        continue

                if m.stage_id in stages and m.block_name in block_names:
                    if hasattr(m, attribute):
                        setattr(m, attribute, attribute_goal_state)
                        count_affected = count_affected + 1
                        if printout: print(f"{attribute} {m.block_name} in {m.stage_id}", f"                module {count_all} of type {type(m)}:")
                    else:
                        print(f"Warning: {attribute} does not exist in {m}")
            else:
                if printout and False: # toggle to view all model for debugging
                    print(f"No {attribute} implemented for module {count_all} of type {type(m)}")
        if printout:
            print(f"SUMMARY: toggle_selective has toggled attribute {attribute} to {attribute_goal_state} for {count_affected} out of {count_all} modules. There were {count_candidates} QConvLayers.\nStages = {stages} and block_names = {block_names}.")
            spared_parts = []
            if spare_bottlenecks:
                spared_parts.append('bottlenecks')
            if spare_attention_qkv:
                spared_parts.append('attention qkv')
            if spare_attention_scaling:
                spared_parts.append('attention scaling')
            if spare_attention_projection:
                spared_parts.append('attention projection')
            print(f"Spared parts: {', '.join(spared_parts)}" if spared_parts else "Did not spare any parts.")
    '''

    def simple_toggle_selective_attribute(
            self, 
            attribute: str,
            attribute_goal_state=True,
            printout=False,
            stages=None, #required: format [string, string, string]
            block_positions=None, #optional: format [int, int, int]
            layer_positions=None, #optional: format [int, int, int]
            ):

        if stages is None or block_positions is None:
            raise NotImplementedError('The backbone format is incompatible with the expected format: stages and block_position must be specified')

        count_all = count_candidates = count_affected = 0
        for m in self.modules():
            count_all = count_all + 1
            if type(m) in [QConvLayer, QConvLayerV2, QLiteMLA]:
                count_candidates = count_candidates + 1
                if m.stage_id in stages:
                    if m.block_position is None or m.block_position in block_positions:
                        if layer_positions is None or m.layer_position in layer_positions:
                            if hasattr(m, attribute):
                                setattr(m, attribute, attribute_goal_state)
                                count_affected = count_affected + 1
                                if printout: print(f"{attribute} == {attribute_goal_state} for {m.block_name}-layer with id {m.stage_id}:{m.block_position}:{m.layer_position}", f"                module {count_all} of type {type(m)}:")
                            else:
                                print(f"Warning: {attribute} does not exist in module: {m}")
                        else:
                            if printout: print(f"Attribute {attribute} on module {m.stage_id}:{m.block_position}:{m.layer_position} was not affected due to layer condition", f"                module {count_all} of type {type(m)}:")
                    else:
                        if printout: print(f"Attribute {attribute} on module {m.stage_id}:{m.block_position}:{m.layer_position} was not affected due to block condition", f"                module {count_all} of type {type(m)}:")
                else:
                    if printout: print(f"Attribute {attribute} on module {m.stage_id}:{m.block_position}:{m.layer_position} was not affected due to stage condition", f"                module {count_all} of type {type(m)}:")

        if printout:
            print(f"SUMMARY: Attribute {attribute} to {attribute_goal_state} for {count_affected} out of {count_all} modules. There were {count_candidates} QConvLayers.\nStages = {stages} and block_positions = {block_positions} and layer_positions = {layer_positions}.")


class EfficientViTSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: EfficientViTSamImageEncoder,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        image_size: tuple[int, int] = (1024, 512),
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                SamResize(self.image_size[1]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                    std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
                ),
                SamPad(self.image_size[1]),
            ]
        )

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: tuple[int, ...],
        original_size: tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_size[0], self.image_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks
    
    ######################################################################
    #       Toggles functions for quantization - early versions          #
    ######################################################################
    '''
    def toggle_calibrate_on(self):
        for m in self.image_encoder.modules():
            if type(m) in [QConvLayer, QConvLayerV2]:
                m.calibrate = True

    def toggle_calibrate_off(self):
        for m in self.image_encoder.modules():
            if type(m) in [QConvLayer, QConvLayerV2]:
                m.calibrate = False

    def toggle_last_calibrate_on(self):
        for m in self.image_encoder.modules():
            if type(m) in [QConvLayer, QConvLayerV2]:
                m.last_calibrate = True

    def toggle_last_calibrate_off(self):
      for m in self.image_encoder.modules():
            if type(m) in [QConvLayer, QConvLayerV2]:
                m.last_calibrate = False
    
    def toggle_quant_on(self):
        for m in self.image_encoder.modules():
            if type(m) in [QConvLayer, QConvLayerV2]:
                m.quant_weights = True

    def toggle_quant_off(self):
        for m in self.image_encoder.modules():
            if type(m) in [QConvLayer, QConvLayerV2]:
                m.quant_weights = False
    
    def toggle_selective_calibrate_on(self, **kwargs):
        self.image_encoder.toggle_selective_attribute(attribute="calibrate", **kwargs,)
        
    def toggle_selective_calibrate_off(self, **kwargs):
        self.image_encoder.toggle_selective_attribute(attribute="calibrate", attribute_goal_state=False, **kwargs,)

    def toggle_selective_last_calibrate_on(self, **kwargs):
        self.image_encoder.toggle_selective_attribute(attribute="last_calibrate", **kwargs,)
    
    def toggle_selective_last_calibrate_off(self, **kwargs):
        self.image_encoder.toggle_selective_attribute(attribute="last_calibrate", attribute_goal_state=False, **kwargs,)

    def toggle_selective_quant_on(self, **kwargs):
        self.image_encoder.toggle_selective_attribute(attribute="quant_weights", **kwargs,)

    def toggle_selective_quant_off(self, **kwargs):
        self.image_encoder.toggle_selective_attribute(attribute="quant_weights", attribute_goal_state=False, **kwargs,)
    '''
    ######################################################################
    #       Toggles functions for quantization - better versions         #
    ######################################################################

    ### Simple versions: expects other backbone formats
    def toggle_selective_calibrate_on(self, **kwargs):
        self.image_encoder.simple_toggle_selective_attribute(attribute="calibrate", **kwargs,)
        
    def toggle_selective_calibrate_off(self, **kwargs):
        self.image_encoder.simple_toggle_selective_attribute(attribute="calibrate", attribute_goal_state=False, **kwargs,)

    def toggle_selective_last_calibrate_on(self, **kwargs):
        self.image_encoder.simple_toggle_selective_attribute(attribute="last_calibrate", **kwargs,)
    
    def toggle_selective_last_calibrate_off(self, **kwargs):
        self.image_encoder.simple_toggle_selective_attribute(attribute="last_calibrate", attribute_goal_state=False, **kwargs,)

    def toggle_selective_quant_weights_on(self, **kwargs):
        self.image_encoder.simple_toggle_selective_attribute(attribute="quant_weights", **kwargs,)

    def toggle_selective_quant_weights_off(self, **kwargs):
        self.image_encoder.simple_toggle_selective_attribute(attribute="quant_weights", attribute_goal_state=False, **kwargs,)

    def toggle_selective_quant_activations_on(self, **kwargs):
        self.image_encoder.simple_toggle_selective_attribute(attribute="quant_activations", **kwargs,)

    def toggle_selective_quant_activations_off(self, **kwargs):
        self.image_encoder.simple_toggle_selective_attribute(attribute="quant_activations", attribute_goal_state=False, **kwargs,)

    def toggle_selective_quant_norms_on(self, **kwargs):
        self.image_encoder.simple_toggle_selective_attribute(attribute="quant_norms", **kwargs,)

    def toggle_selective_quant_norms_off(self, **kwargs):
        self.image_encoder.simple_toggle_selective_attribute(attribute="quant_norms", attribute_goal_state=False, **kwargs,)

    ### statistics
    def toggle_monitor_distributions_on(self, **kwargs):
        for m in self.image_encoder.modules():
            if type(m) in [QConvLayer, QConvLayerV2]:
                m.monitor_distributions = True

    def toggle_monitor_distributions_off(self, **kwargs):
        for m in self.image_encoder.modules():
            if type(m) in [QConvLayer, QConvLayerV2]:
                m.monitor_distributions = False

    def get_number_of_quantized_params(self):
        affected = 0
        unaffected = 0
        for m in self.image_encoder.modules():
            if type(m) in [QConvLayer, QConvLayerV2]:
                if m.quant_weights:
                    affected = affected + m.parameter_count()
                else:
                    unaffected = unaffected + m.parameter_count()
        return affected, unaffected

    def print_named_parameters(self):
        for name, param in self.image_encoder.named_parameters():
            print("param name:", name)
            print("param.size():", param.size())

    def print_some_statistics(self):
        counter = 0
        for m in self.image_encoder.modules():
            if type(m) in [QConvLayer, QConvLayerV2]:
                observer = m.weight_observer
                if observer.stage_id == 'stage2':
                    counter += 1
                    if counter == 2:                    
                        print(f"Observer with info: {observer.stage_id}, {observer.block_name}, {observer.operation_type}")
                        tensor = observer.stored_weight_tensor
                        tensor = tensor.detach()
                        tensor = tensor.cpu() #torch.histogram is not implemented on CUDA backend.
                        # tensor = tensor.cuda()
                        
                        '''The shape of the weight tensor corresponds to (out_channels, in_channels, kernel_height, kernel_width).'''
                        print(f"stored weight tensor: {tensor.size()}"
                        f"Mean: {tensor.mean().item()}\n"
                        f"Std: {tensor.std().item()}\n"
                        f"Min: {tensor.min().item()}\n"
                        f"Max: {tensor.max().item()}")

                        hist_values, bin_edges = torch.histogram(tensor, density=True)

                        plt.bar(bin_edges[:-1], hist_values, width = 0.1)
                        plt.title(f"Weights of {observer.stage_id}, {observer.block_name}, {observer.operation_type}, second block \n all values of tensor with shape {tensor.size()}")
                       
                        plt.axvline(tensor.min().item(), color='r', linestyle='dotted', linewidth=1)
                        plt.axvline(tensor.max().item(), color='g', linestyle='dotted', linewidth=1)
                        plt.axvline(torch.quantile(tensor, 0.5).item(), color='b', linestyle='dotted', linewidth=1)  # 50th percentile (median)
                        plt.xlabel("Weight value")
                        plt.ylabel("Normalized number of occurances")

                        plt.savefig(f'./plots/My_first_histogram.png')
                        plt.close()

                         # Reshape tensor to 2D, with second dimension being the flattened kernel
                        tensor_2d = tensor.view(tensor.shape[0], -1)
                        # Select the first 12 channels
                        tensor_2d = tensor_2d[:12]

                        # Create boxplot
                        plt.boxplot(tensor_2d, vert=True, patch_artist=True)
                        plt.title(f"Weights of 12 channels of {observer.stage_id}, {observer.block_name}, {observer.operation_type}, second block \n all values of tensor with shape {tensor.size()}")
                        plt.xlabel("Channel number (output)")
                        plt.ylabel("Weight value")

                        #plt.xlim(0, tensor_2d.shape[0] + 5)  # adjust '5' as needed

                        #plt.text(tensor_2d.shape[0] + 1, tensor_2d.min(), 'Box: Interquartile Range (IQR)\nLine in Box: Median\nWhiskers: Range within 1.5*IQR\nCircles: Outliers', verticalalignment='bottom')

                        plt.savefig(f'./plots/My_first_boxplot.png')
                        plt.close()

        '''
        Observer with info: stage2, fmb, conv_weight
        stored weight tensor: torch.Size([1024, 64, 3, 3])
        Observer with info: stage2, fmb, conv_weight
        stored weight tensor: torch.Size([128, 1024, 1, 1])
        Observer with info: stage2, fmb, conv_weight
        stored weight tensor: torch.Size([512, 128, 3, 3])
        Observer with info: stage2, fmb, conv_weight
        stored weight tensor: torch.Size([128, 512, 1, 1])'''


class EfficientViTSamPredictor:
    def __init__(self, sam_model: EfficientViTSam) -> None:
        self.model = sam_model
        self.reset_image()

    @property
    def transform(self):
        return self

    @property
    def device(self):
        return get_device(self.model)

    def reset_image(self) -> None:
        self.is_image_set = False
        self.features = None
        self.original_size = None
        self.input_size = None

    def apply_coords(self, coords: np.ndarray, im_size=None) -> np.ndarray:
        old_h, old_w = self.original_size
        new_h, new_w = self.input_size
        coords = copy.deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, im_size=None) -> np.ndarray:
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2))
        return boxes.reshape(-1, 4)

    @torch.inference_mode()
    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        self.reset_image()

        self.original_size = image.shape[:2]
        self.input_size = ResizeLongestSide.get_preprocess_shape(
            *self.original_size, long_side_length=self.model.image_size[0]
        )

        torch_data = self.model.transform(image).unsqueeze(dim=0).to(get_device(self.model))
        self.features = self.model.image_encoder(torch_data)
        self.is_image_set = True

    def predict(
        self,
        point_coords: np.ndarray or None = None,
        point_labels: np.ndarray or None = None,
        box: np.ndarray or None = None,
        mask_input: np.ndarray or None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        device = get_device(self.model)
        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert point_labels is not None, "point_labels must be supplied if point_coords is supplied."
            point_coords = self.apply_coords(point_coords)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.apply_boxes(box)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        masks = masks[0].detach().cpu().numpy()
        iou_predictions = iou_predictions[0].detach().cpu().numpy()
        low_res_masks = low_res_masks[0].detach().cpu().numpy()
        return masks, iou_predictions, low_res_masks

    @torch.inference_mode()
    def predict_torch(
        self,
        point_coords: torch.Tensor or None = None,
        point_labels: torch.Tensor or None = None,
        boxes: torch.Tensor or None = None,
        mask_input: torch.Tensor or None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks


class EfficientViTSamAutomaticMaskGenerator(SamAutomaticMaskGenerator):
    def __init__(
        self,
        model: EfficientViTSam,
        points_per_side: int or None = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: list[np.ndarray] or None = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> None:
        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = EfficientViTSamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode


def build_efficientvit_sam(image_encoder: EfficientViTSamImageEncoder, image_size: int) -> EfficientViTSam:
    return EfficientViTSam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        image_size=(1024, image_size),
    )


def efficientvit_sam_l0(image_size: int = 512, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l0

    backbone = efficientvit_backbone_l0(**kwargs)

    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=4,
        expand_ratio=1,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_l1(image_size: int = 512, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l1

    backbone = efficientvit_backbone_l1(**kwargs)

    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=8,
        expand_ratio=1,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_l2(image_size: int = 512, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l2

    backbone = efficientvit_backbone_l2(**kwargs)

    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=12,
        expand_ratio=1,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_xl0(image_size: int = 1024, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import EfficientViTLargeBackbone

    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512, 1024],
        depth_list=[0, 1, 1, 2, 3, 3],
        block_list=["res", "fmb", "fmb", "fmb", "att@3", "att@3"],
        expand_list=[1, 4, 4, 4, 4, 6],
        fewer_norm_list=[False, False, False, False, True, True],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )

    neck = SamNeck(
        fid_list=["stage5", "stage4", "stage3"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=6,
        expand_ratio=4,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_xl1(image_size: int = 1024, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import EfficientViTLargeBackbone

    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512, 1024],
        depth_list=[1, 2, 2, 4, 6, 6],
        block_list=["res", "fmb", "fmb", "fmb", "att@3", "att@3"],
        expand_list=[1, 4, 4, 4, 4, 6],
        fewer_norm_list=[False, False, False, False, True, True],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )

    neck = SamNeck(
        fid_list=["stage5", "stage4", "stage3"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=12,
        expand_ratio=4,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


######################################################################
#                         Quantized builders                         #
######################################################################

def efficientvit_sam_l0_quant(image_size: int = 512, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l0_quant     

    backbone = efficientvit_backbone_l0_quant(**kwargs)

    neck = QSamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=4,
        expand_ratio=1,
        middle_op="fmb",
        **build_kwargs_from_config(kwargs, QSamNeck),
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_l1_quant(image_size: int = 512, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l1_quant

    backbone = efficientvit_backbone_l1_quant(**kwargs)

    neck = QSamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=8,
        expand_ratio=1,
        middle_op="fmb",
        **build_kwargs_from_config(kwargs, QSamNeck),
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_l2_quant(image_size: int = 512, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l2_quant

    backbone = efficientvit_backbone_l2_quant(**kwargs)

    neck = QSamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=12,
        expand_ratio=1,
        middle_op="fmb",
        **build_kwargs_from_config(kwargs, QSamNeck),
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_xl0_quant(image_size: int = 1024, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import EfficientViTLargeBackboneQuant

    backbone = EfficientViTLargeBackboneQuant(
        width_list=[32, 64, 128, 256, 512, 1024],
        depth_list=[0, 1, 1, 2, 3, 3],
        block_list=["res", "fmb", "fmb", "fmb", "att@3", "att@3"],
        expand_list=[1, 4, 4, 4, 4, 6],
        fewer_norm_list=[False, False, False, False, True, True],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackboneQuant),
    )

    neck = QSamNeck(
        fid_list=["stage5", "stage4", "stage3"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=6,
        expand_ratio=4,
        middle_op="fmb",
        **build_kwargs_from_config(kwargs, QSamNeck),        
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_xl1_quant(image_size: int = 1024, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import EfficientViTLargeBackboneQuant

    backbone = EfficientViTLargeBackboneQuant(
        width_list=[32, 64, 128, 256, 512, 1024],
        depth_list=[1, 2, 2, 4, 6, 6],
        block_list=["res", "fmb", "fmb", "fmb", "att@3", "att@3"],
        expand_list=[1, 4, 4, 4, 4, 6],
        fewer_norm_list=[False, False, False, False, True, True],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackboneQuant),
    )

    neck = QSamNeck(
        fid_list=["stage5", "stage4", "stage3"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=12,
        expand_ratio=4,
        middle_op="fmb",
        **build_kwargs_from_config(kwargs, QSamNeck),
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)
