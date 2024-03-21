# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from efficientvit.models.nn.act import build_act
from efficientvit.models.nn.norm import build_norm
from efficientvit.models.utils import get_same_padding, list_sum, resize, val2list, val2tuple

from efficientvit.models.ptq import BIT_TYPE_DICT
from efficientvit.models.ptq.observer import build_observer
from efficientvit.models.ptq.quantizer import build_quantizer

__all__ = [
    ## basic layers ##
    "ConvLayer",
    "UpSampleLayer",
    "LinearLayer",
    "IdentityLayer",
    ## basic blocks ##
    "DSConv",
    "MBConv",
    "FusedMBConv",
    "ResBlock",
    "LiteMLA",
    "EfficientViTBlock",
    ## functional blocks ##
    "ResidualBlock",
    "DAGBlock",
    "OpSequential",
    ## quantized basic layers ##
    "QConvLayer",
    "QConvLayerV2", # special inheritance version
    "QUpSampleLayer",
    "QLinearLayer",
    "QIdentityLayer",
    ## quantized basic blocks ##
    "QDSConv",
    "QMBConv",
    "QFusedMBConv",
    "QResBlock",
    "QLiteMLA",
    "QEfficientViTBlock",
    ## there are no quantized functional blocks ##
]


#################################################################################
#                             Basic Layers                                      #
#################################################################################


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x
        

class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode="bicubic",
        size: int or tuple[int, int] or list[int] or None = None,
        factor=2,
        align_corners=False,
    ):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.size is not None and tuple(x.shape[-2:]) == self.size) or self.factor == 1:
            return x
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout=0,
        norm=None,
        act_func=None,
    ):
        super(LinearLayer, self).__init__()

        self.dropout = nn.Dropout(dropout, inplace=False) if dropout > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, num_features=out_features)
        self.act = build_act(act_func)

    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


#################################################################################
#                             Basic Blocks                                      #
#################################################################################


class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super(DSConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
    ):
        super(MBConv, self).__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

# used in the Large backbone, and in SAM
class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x

# used for SAM - it's a block from ResNet34
class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int or None = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        super(LiteMLA, self).__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    @autocast(enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        qkv = torch.transpose(qkv, -1, -2)
        q, k, v = (
            qkv[..., 0 : self.dim],
            qkv[..., self.dim : 2 * self.dim],
            qkv[..., 2 * self.dim :],
        )

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)

        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        out = self.relu_linear_att(multi_scale_qkv)
        out = self.proj(out)

        return out

# wraps LiteMLA
class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        scales=(5,),
        norm="bn2d",
        act_func="hswish",
    ):
        super(EfficientViTBlock, self).__init__()
        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
                scales=scales, # act func never overriden!
            ),
            IdentityLayer(),
        )
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x


#################################################################################
#                             Functional Blocks                                 #
#################################################################################

# Wrapper of other blocks. Just a residual shortcut if called with identityLayer
class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: nn.Module or None,
        shortcut: nn.Module or None,
        post_act=None,
        pre_norm: nn.Module or None = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res

# Concat and addition operations - used for SAM
class DAGBlock(nn.Module):
    def __init__(
        self,
        inputs: dict[str, nn.Module],
        merge: str,
        post_input: nn.Module or None,
        middle: nn.Module,
        outputs: dict[str, nn.Module],
    ):
        super(DAGBlock, self).__init__()

        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge = merge
        self.post_input = post_input

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = torch.concat(feat, dim=1)
        else:
            raise NotImplementedError
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict

# A pipeline, or a sequence of operations.
class OpSequential(nn.Module):
    def __init__(self, op_list: list[nn.Module or None]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list: # each op is a layer in a neural net
            x = op(x)           # passes x through each layer
        return x                # returns the final output


#################################################################################
#                          Quantized Basic Layers                               #
#################################################################################

# Implementaiton inspired by QConv2d from FQ-ViT/models/ptq/layers.py    
'''
This one implements nn.Module, while FQViT implements nn.Conv2d. Both should work.
QConvLayer now manually assigns self.conv = nn.Conv2d to use the convolution implementation from torch.nn.
This is why ours doesn't pass any arguments to super.__init__(), while FQViT does.
Weights are stored in self.conv.weight, not self.weight
'''
class QConvLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            stride=1,
            dilation=1,
            groups=1,
            use_bias=False,
            dropout=0,
            norm="bn2d",
            act_func="relu",
            # custom arguments
            quant=False,
            calibrate=False,
            last_calibrate=False,
            bit_type=BIT_TYPE_DICT['int8'],
            calibration_mode='layer_wise',
            observer_str='minmax',
            quantizer_str='uniform',
            stage_id='unknown',
            block_name='independent',
            block_is_bottleneck=False,
            conv_is_attention_qkv=False,
            conv_is_attention_scaling=False,
            conv_is_attention_projection=False,
        ):
        super().__init__()
        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels, # the self.conv.weight tensor gets shape: (out_channels, in_channels, kernel_height, kernel_width). It is named .conv. beacuse of this attribute
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels) # builds nn.Module if not None
        self.act = build_act(act_func)  # builds nn.Module if not None

        # Custom arguments
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str
        self.module_type = 'conv_weight'
        self.stage_id = stage_id
        self.block_name = block_name
        self.block_is_bottleneck = block_is_bottleneck
        self.conv_is_attention_qkv = conv_is_attention_qkv
        self.conv_is_attention_scaling = conv_is_attention_scaling
        self.conv_is_attention_projection = conv_is_attention_projection

        observer_object = build_observer(self.observer_str, self.module_type, 
                                       self.bit_type, self.calibration_mode) # in FQ-ViT, this is saved as self.observer for no reason
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                          observer_object, self.module_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # calibrate
        if self.calibrate:
            #print(f"Forward method called on device {x.device}. calibrate = {self.calibrate}, last_calib = {self.last_calibrate}")
            self.quantizer.observer.update(self.conv.weight) # for all batches of calibration data: update statistics
            #print("Calibration batch processed in QConv2d")
            if self.last_calibrate:                          # after the last batch, fetch S and Z of the quantizer
                self.quantizer.update_quantization_params(x) # maybe x is not needed as argument
                #print("Calibration finished in QConv2d.")

        # dropout
        if self.dropout is not None:
            x = self.dropout(x)

        # quantization
        if self.quant:
            # quant + dequant the weights
            w = self.quantizer(self.conv.weight)
            # passing the parameters from self.conv to F.conv2d
            x = F.conv2d(x, w, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        else:
            x = self.conv(x)
         
        # normalization
        if self.norm:
            x = self.norm(x) # calls the activation function
        
        # activation
        if self.act:
            x = self.act(x)

        return x
    
    def parameter_count(self):
        num_weights = self.conv.weight.numel()
        num_biases = self.conv.bias.numel() if self.conv.bias is not None else 0
        return num_weights + num_biases


class QConvLayerV2(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            stride=1,
            dilation=1,
            groups=1,
            use_bias=False,
            dropout=0,
            norm="bn2d",
            act_func="relu",
            # custom arguments
            quant=False,
            calibrate=False,
            last_calibrate=False,
            bit_type=BIT_TYPE_DICT['int8'],
            calibration_mode='layer_wise',
            observer_str='minmax',
            quantizer_str='uniform',
            stage_id='unknown',
            block_name='independent',
            block_is_bottleneck=False,
            conv_is_attention_qkv=False,
            conv_is_attention_scaling=False,
            conv_is_attention_projection=False,
        ):
        
        padding = get_same_padding(kernel_size)
        padding *= dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        '''self.conv = nn.Conv2d(
            in_channels, # the self.conv.weight tensor gets shape: (out_channels, in_channels, kernel_height, kernel_width). It is named .conv. beacuse of this attribute
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )'''
        self.norm = build_norm(norm, num_features=out_channels) # builds nn.Module if not None
        self.act = build_act(act_func)  # builds nn.Module if not None

        # Custom arguments
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str
        self.module_type = 'conv_weight'
        self.stage_id = stage_id
        self.block_name = block_name
        self.block_is_bottleneck = block_is_bottleneck
        self.conv_is_attention_qkv = conv_is_attention_qkv
        self.conv_is_attention_scaling = conv_is_attention_scaling
        self.conv_is_attention_projection = conv_is_attention_projection

        observer_object = build_observer(self.observer_str, self.module_type, 
                                       self.bit_type, self.calibration_mode) # in FQ-ViT, this is saved as self.observer for no reason
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                          observer_object, self.module_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # calibrate
        if self.calibrate:
            #print(f"Forward method called on device {x.device}. calibrate = {self.calibrate}, last_calib = {self.last_calibrate}")
            self.quantizer.observer.update(self.weight) # for all batches of calibration data: update statistics
            #print("Calibration batch processed in QConv2d")
            if self.last_calibrate:                          # after the last batch, fetch S and Z of the quantizer
                self.quantizer.update_quantization_params(x) # maybe x is not needed as argument
                #print("Calibration finished in QConv2d.")

        # dropout
        if self.dropout is not None:
            x = self.dropout(x)

        # quantization
        if self.quant:
            # quant + dequant the weights
            w = self.quantizer(self.weight)
            # passing the parameters from self.conv to F.conv2d
            x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            x = super().forward(x)  # Conv2d is inherited in V2, instead of an attribute
         
        # normalization
        if self.norm:
            x = self.norm(x) # calls the activation function
        
        # activation
        if self.act:
            x = self.act(x)

        return x

    def parameter_count(self):
        num_weights = self.weight.numel()
        num_biases = self.bias.numel() if self.bias is not None else 0
        return num_weights + num_biases


# Not used for anything??
class QUpSampleLayer(nn.Module):
    def __init__(
        self,
        mode="bicubic",
        size: int or tuple[int, int] or list[int] or None = None,
        factor=2,
        align_corners=False,
    ):
        super().__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.size is not None and tuple(x.shape[-2:]) == self.size) or self.factor == 1:
            return x
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class QLinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout=0,
        norm=None,
        act_func=None,
        # custom arguments
            # custom arguments
            quant=False,
            calibrate=False,
            last_calibrate=False,
            bit_type=BIT_TYPE_DICT['int8'],
            calibration_mode='layer_wise',
            observer_str='minmax',
            quantizer_str='uniform',
            stage_id='unknown',
            block_name='independent',
            block_is_bottleneck=False,
        ):
        super().__init__()

        self.dropout = nn.Dropout(dropout, inplace=False) if dropout > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, num_features=out_features)
        self.act = build_act(act_func)

        # Custom arguments
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str
        self.module_type = 'linear_weight'
        self.state_id=stage_id
        self.block_name=block_name
        self.block_is_bottleneck=block_is_bottleneck
        observer_object = build_observer(self.observer_str, self.module_type, 
                                       self.bit_type, self.calibration_mode) # in FQ-ViT, this is saved as self.observer for no reason
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                          observer_object, self.module_type)

    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if self.dropout:
            x = self.dropout(x)
        # calibration
        if self.calibrate:
            self.quantizer.observer.update(self.linear.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        
        # quantization
        if self.quant:
            w = self.quantizer(self.linear.weight)
            x = F.linear(x, w, self.linear.bias)
        else:        
            x = self.linear(x)

        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


# not implemented
class QIdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x



#################################################################################
#                         Quantized Basic Blocks                                #
#################################################################################


# Wraps QConvLayer twice. No other modifications
class QDSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
        **kwargs
        ):
        super().__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = QConvLayer(      
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
            **kwargs,
            )
        self.point_conv = QConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

# Wraps QConvLayer three times. No other modifications
class QMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
        **kwargs, # config arguments
    ):
        super().__init__()
        
        # if apssed a singular norm,  bias or act, convert into tuple of length 3
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = QConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
            **kwargs, # config arguments
        )
        self.depth_conv = QConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
            **kwargs, # config arguments
        )
        self.point_conv = QConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
            **kwargs, # config arguments
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

# Wraps QConvLayer twice. No other modifications. Is only used in Large models!
class QFusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
        **kwargs, # config arguments
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = QConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
            **kwargs, # config arguments
        )
        self.point_conv = QConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
            **kwargs, # config arguments
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x

# Wraps QConvLayer twice. No other modifications. Is only used in SAM models
class QResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
        **kwargs, # config arguments
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.conv1 = QConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
            **kwargs, # config arguments
        )
        self.conv2 = QConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
            **kwargs, # config arguments
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# Not used in backbone.py directly, only via QEfficientVitBlock
# Not Quantized yet!
class QLiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int or None = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: tuple[int, ...] = (5,), # hinting that it should be a tuple of integers, and initializing it with a tuple containing the single integer 5. XL-models have (3,) for some layers. Large models have (5,)
        eps=1.0e-15,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = QConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0], # False
            norm=norm[0],         # override to b2nd
            act_func=act_func[0], # override to Gelu
            conv_is_attention_qkv=True,
            **kwargs,
        )

        '''

        The pretrained weights have state keys on the form aggreg.0.0.weights, because here nn.Conv2D was used directly unlike elsewhere in EfficientViT
        Using QConvLayer(nn.Module) causes state dict keys on the form aggreg.0.0.conv.weights, because the nn.Conv2D is an attribute named conv in the module.
        One solution is to rebuild QConvLayer to inherit from nn.Conv2D instead of nn.Module, but this might alter other state keys in the model.'''
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    QConvLayerV2(
                        in_channels = 3 * total_dim,
                        out_channels = 3 * total_dim,
                        kernel_size=scale,
                        # padding=get_same_padding(scale), handled inside QConvLayer
                        groups = 3 * total_dim,
                        use_bias = use_bias[0],
                        norm=None,
                        act_func=None,
                        conv_is_attention_scaling=True,
                        **kwargs,
                    ),
                    QConvLayerV2(
                        in_channels = 3 * total_dim,
                        out_channels = 3 * total_dim,
                        kernel_size = 1,
                        groups = 3 * heads,
                        use_bias = use_bias[0],
                        norm=None,
                        act_func=None,
                        conv_is_attention_scaling=True,
                        **kwargs,
                    )
                )
                for scale in scales
            ]
        )

        '''         QConvLayer(
                    in_channels = 3 * total_dim,
                    out_channels = 3 * total_dim,
                    kernel_size = 1,
                    groups = 3 * heads,
                    use_bias = use_bias[0],
                    norm=None,
                    act_func=None,
                    conv_is_attention_scaling=True,
                    **kwargs,
                )'''
    
        '''        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        '''

        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = QConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],         # override to bn2d
            act_func=act_func[1], # override to gelu
            conv_is_attention_projection=True,
            **kwargs,
        )

    @autocast(enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        qkv = torch.transpose(qkv, -1, -2)
        q, k, v = (
            qkv[..., 0 : self.dim],
            qkv[..., self.dim : 2 * self.dim],
            qkv[..., 2 * self.dim :],
        )

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)

        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        out = self.relu_linear_att(multi_scale_qkv)
        out = self.proj(out)

        return out


class QEfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        scales=(5,),            #XL-models have (3,) for some layers. Large models have (5,)
        norm="bn2d",
        act_func="hswish",
        **kwargs,
    ):
        super().__init__()
        self.context_module = ResidualBlock(
            QLiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
                scales=scales,
                **kwargs,
            ),
            IdentityLayer(),
        )
        local_module = QMBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
            **kwargs,
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x


#################################################################################
#                   There are no Quantized Functional Blocks                    #
#################################################################################

