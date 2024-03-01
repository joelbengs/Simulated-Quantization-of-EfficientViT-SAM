# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import torch
import torch.nn as nn

from efficientvit.models.efficientvit.backbone import EfficientViTBackbone, EfficientViTLargeBackbone, EfficientViTBackboneQuant
from efficientvit.models.nn import ConvLayer, LinearLayer, OpSequential
from efficientvit.models.utils import build_kwargs_from_config

# imports of quantized layers needed for the toggle functions
from efficientvit.models.nn import QConvLayer


__all__ = [
    "EfficientViTCls",
    ######################
    "efficientvit_cls_b0",
    "efficientvit_cls_b1",
    "efficientvit_cls_b2",
    "efficientvit_cls_b3",
    ######################
    "efficientvit_cls_l1",
    "efficientvit_cls_l2",
    "efficientvit_cls_l3",
    ## quantized models ##,
    "efficientvit_cls_b1_quant",
]


class ClsHead(OpSequential):
    def __init__(
        self,
        in_channels: int,
        width_list: list[int],
        n_classes=1000,
        dropout=0.0,
        norm="bn2d",
        act_func="hswish",
        fid="stage_final", # question: where in the code are they upscaling and concatenating intermediate feature maps?
    ):
        ops = [
            ConvLayer(in_channels, width_list[0], 1, norm=norm, act_func=act_func),
            nn.AdaptiveAvgPool2d(output_size=1),
            LinearLayer(width_list[0], width_list[1], False, norm="ln", act_func=act_func),
            LinearLayer(width_list[1], n_classes, True, dropout, None, None),
        ]
        super().__init__(ops) # another way of making the above layers callable by pytorch in the forward pass.

        self.fid = fid

    def forward(self, feed_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        x = feed_dict[self.fid] #hardcode to extract just the final feature map from the backbone.
        return OpSequential.forward(self, x) # passes x through itself, using the inherited forward method.

# The complete model is of this class
class EfficientViTCls(nn.Module):
    def __init__(self, 
                 backbone: EfficientViTBackbone or EfficientViTLargeBackbone or EfficientViTBackboneQuant, 
                 head: ClsHead,
                 
                 
                 
                 ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feed_dict = self.backbone(x)     # backbone inference.     backbone is instance of nn.Module: this -> nn.Module.__call__() --> backbone.forward()
        output = self.head(feed_dict)    # head inference.         head is instance of ClsHead, which is instance of OpSequential (special function) which in turn is instance of nn.Module.
        return output
    

    ######################################################################
    #                     quantization and calibration                   #
    #                    (shifts attributes true/false)                  #
    ######################################################################

    def toggle_calibrate_on(self):
        for m in self.modules():
            if type(m) in [QConvLayer]:
                m.calibrate = True
        print("Sucessfully reached toggled calibrate on")

    def toggle_calibrate_off(self):
        for m in self.modules():
            if type(m) in [QConvLayer]:
                m.calibrate = False

    def toggle_last_calibrate_on(self):
        for m in self.modules():
            if type(m) in [QConvLayer]:
                m.last_calibrate = True

    def toggle_last_calibrate_off(self):
      for m in self.modules():
            if type(m) in [QConvLayer]:
                m.last_calibrate = False
    
    def toggle_quant_on(self):
        for m in self.modules():
            if type(m) in [QConvLayer]:
                m.quant = True
            #if self.cfg.INT_NORM:
             #   if type(m) in [QIntLayerNorm]:
              #      m.mode = 'int'
    
    def toggle_quant_off(self):
        for m in self.modules():
            if type(m) in [QConvLayer]:
                m.quant = True
            #if self.cfg.INT_NORM:
             #   if type(m) in [QIntLayerNorm]:
              #      m.mode = 'int'



def efficientvit_cls_b0(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b0

    backbone = efficientvit_backbone_b0(**kwargs)

    head = ClsHead(
        in_channels=128,
        width_list=[1024, 1280],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def efficientvit_cls_b1(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b1

    backbone = efficientvit_backbone_b1(**kwargs)

    head = ClsHead(
        in_channels=256,
        width_list=[1536, 1600],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def efficientvit_cls_b2(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b2

    backbone = efficientvit_backbone_b2(**kwargs)

    head = ClsHead(
        in_channels=384,
        width_list=[2304, 2560],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def efficientvit_cls_b3(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b3

    backbone = efficientvit_backbone_b3(**kwargs)

    head = ClsHead(
        in_channels=512,
        width_list=[2304, 2560],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def efficientvit_cls_l1(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l1

    backbone = efficientvit_backbone_l1(**kwargs)

    head = ClsHead(
        in_channels=512,
        width_list=[3072, 3200],
        act_func="gelu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def efficientvit_cls_l2(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l2

    backbone = efficientvit_backbone_l2(**kwargs)

    head = ClsHead(
        in_channels=512,
        width_list=[3072, 3200],
        act_func="gelu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def efficientvit_cls_l3(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l3

    backbone = efficientvit_backbone_l3(**kwargs)

    head = ClsHead(
        in_channels=1024,
        width_list=[6144, 6400],
        act_func="gelu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def efficientvit_cls_b1_quant(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b1_quant

    # Step 3: call for a quantized backbone
    backbone = efficientvit_backbone_b1_quant(**kwargs)

    head = ClsHead(
        in_channels=256,
        width_list=[1536, 1600],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model
