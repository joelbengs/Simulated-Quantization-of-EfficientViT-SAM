# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023
# Modified by Joel Bengs on 2024-06-11 under Apache-2.0 license
# Changes made:
# - Implemented simulation of mixed-precision quantization to further accelerate EfficientViT-SAM


import torch
import torch.nn as nn

from efficientvit.models.nn import (
    ConvLayer,
    DSConv,
    EfficientViTBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResBlock,
    ResidualBlock,
    ## quantized modules ##
    ## Quantized ops ##
    QConvLayer,
    QDSConv,
    QMBConv,
    QFusedMBConv,
    QResBlock,
    QLiteMLA,
    QEfficientViTBlock,
)
from efficientvit.models.utils import build_kwargs_from_config

__all__ = [
    "EfficientViTBackbone",
    "efficientvit_backbone_b0",
    "efficientvit_backbone_b1",
    "efficientvit_backbone_b2",
    "efficientvit_backbone_b3",
    "EfficientViTLargeBackbone",
    "efficientvit_backbone_l0",
    "efficientvit_backbone_l1",
    "efficientvit_backbone_l2",
    "efficientvit_backbone_l3",
    #### quantized backbones for CLS ####
    "EfficientViTBackboneQuant",
    "efficientvit_backbone_b1_quant",
    #### quantized backbones for SAM ####
    "EfficientViTLargeBackboneQuant",
    "efficientvit_backbone_l0_quant",
    "efficientvit_backbone_l1_quant",
    "efficientvit_backbone_l2_quant",
    "efficientvit_backbone_l3_quant", # never used
]

# width_list=[
            # # kernels in input Conv, 
            # # input & output channels in each MBConv in stage 1 ,
            # # input & output channels in each MBConv in stage 2,
            # # input & output channels in each module in stage 3,
            # # input & output channels in each module in stage 4,
            # ]

# depth_list=[
            # # DSConv in input stem, 
            # # MBConv in stage 1 ,
            # # MBConv in stage 2,
            # # novel EfficientViT modules in stage 4,
            # # novel EfficientViT modules in stage 4,
            # ]

class EfficientViTBackbone(nn.Module):
    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        in_channels=3,
        dim=32,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
    ) -> None:
        super().__init__()

        self.width_list = []
        # input stem
        self.input_stem = [
            ConvLayer(
                in_channels=3,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)

        # stages
        self.stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output_dict = {"input": x}
        output_dict["stage0"] = x = self.input_stem(x)
        for stage_id, stage in enumerate(self.stages, 1):
            output_dict["stage%d" % stage_id] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict


def efficientvit_backbone_b0(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b1(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[1, 2, 3, 3, 4],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b2(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b3(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


class EfficientViTLargeBackbone(nn.Module):
    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        block_list: list[str] or None = None,
        expand_list: list[float] or None = None,
        fewer_norm_list: list[bool] or None = None,
        in_channels=3,
        qkv_dim=32,
        norm="bn2d",
        act_func="gelu",
    ) -> None:
        super().__init__()
        block_list = block_list or ["res", "fmb", "fmb", "mb", "att"]
        expand_list = expand_list or [1, 4, 4, 4, 6]
        fewer_norm_list = fewer_norm_list or [False, False, False, True, True]

        self.width_list = []
        self.stages = []
        # stage 0
        stage0 = [
            ConvLayer(
                in_channels=3,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                block=block_list[0],
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=expand_list[0],
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[0],
            )
            stage0.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.stages.append(OpSequential(stage0))
        self.width_list.append(in_channels)

        for stage_id, (w, d) in enumerate(zip(width_list[1:], depth_list[1:]), start=1):
            stage = []
            block = self.build_local_block(
                block="mb" if block_list[stage_id] not in ["mb", "fmb"] else block_list[stage_id],
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_list[stage_id] * 4,
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[stage_id],
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                if block_list[stage_id].startswith("att"):
                    stage.append(
                        EfficientViTBlock(
                            in_channels=in_channels,
                            dim=qkv_dim,
                            expand_ratio=expand_list[stage_id],
                            scales=(3,) if block_list[stage_id] == "att@3" else (5,),
                            norm=norm,
                            act_func=act_func,
                        )
                    )
                else:
                    block = self.build_local_block(
                        block=block_list[stage_id],
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=1,
                        expand_ratio=expand_list[stage_id],
                        norm=norm,
                        act_func=act_func,
                        fewer_norm=fewer_norm_list[stage_id],
                    )
                    block = ResidualBlock(block, IdentityLayer())
                    stage.append(block)
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        block: str,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if block == "res":
            block = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif block == "fmb":
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif block == "mb":
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        else:
            raise ValueError(block)
        return block

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output_dict = {"input": x}
        for stage_id, stage in enumerate(self.stages):
            output_dict["stage%d" % stage_id] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict


def efficientvit_backbone_l0(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 4, 4],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l1(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 6, 6],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l2(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 2, 2, 8, 8],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l3(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[64, 128, 256, 512, 1024],
        depth_list=[1, 2, 2, 8, 8],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone

######################################################################
#                         Quantized classification                   #
######################################################################

class EfficientViTBackboneQuant(nn.Module):
    def __init__(
        self,
        width_list: list[int],  
        depth_list: list[int],  
        in_channels=3,          # RGB
        dim=32,                 # dimension of EViT module, overridden to 16 in model b1
        expand_ratio=4,         # The ratio by which to expand the channels in a block if = 1, all MBConvs are turn into DSConv. 
        norm="bn2d",            # Normalization
        act_func="hswish",      # activation function
        config=None,            # configurations built by quant_config.py, using parsed arguments
    ) -> None:
        super().__init__()

        self.width_list = []    # for logging, doesn't affect backbone building.
        self.config=config

        #################################################################################
        #                       Input Stem of EfficientViT                              #
        #             Convolution + Depthwise Seperable Convolution (DSConv)            #
        #################################################################################

        # Part 1: INPUT STEAM = 1 ConvLayer + several DSConv ResidualBlocks, all wrapped in OpSequential
        #
        # width_list=[16, _, _, _, _], depth_list=[1, _, _, _, _], dim=16
        # -->
        # A ConvLayer with 16 output channels (i.e. 16 kernels). This layer has 3 input channels by default. The stride is 2, and the normalization and activation functions are determined by build_kwargs_from_config(kwargs, EfficientViTBackbone).
        # Then, a number of residual blocks equal to the first element of depth_list (which is 1 in this case) are added. Each residual block consists of a local block and an IdentityLayer. The local block is created by the build_local_block method with 16 input channels, 16 output channels, a stride of 1, and an expand ratio of 1. The normalization and activation functions are again determined by build_kwargs_from_config

        # 1 ConvLayer --> QConvLayer
        self.input_stem = [
            QConvLayer(
                in_channels=3,              # RGB input
                out_channels=width_list[0], # width_list=[16, _, _, _, _] --> 16 kernels
                stride=2,                   # Stride 2
                norm=norm,
                act_func=act_func,        
            )
        ]

        # Residual blocks - specifically depth[0] DSConvs
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,             # forces a DSConv instead of MBConv
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        # save the channel depth at output of the stem
        in_channels = width_list[0]
        self.width_list.append(in_channels)

        # Wrap in OpSequential, which enables PyTorch to function as should
        self.input_stem = OpSequential(self.input_stem)


        #################################################################################
        #                   Early stages of EfficientViT (stage 1 and 2)                #
        #                  Mobile Inverted Bottleneck Convolutions (MBConv)             #
        #################################################################################
        
        # A stage = a sequence of blocks.
        # A block = a local_block wrapped in a ResidualBlock.
        self.stages = []
        
        # Loops over second and third elements of width and depth.
        #width_list=[-, 32, 64, -, -]
        #depth_list=[-, 2, 3, -, -]
        for w, d in zip(width_list[1:3], depth_list[1:3]): 
            stage = []
            for i in range(d):                      # each stage has d blocks
                stride = 2 if i == 0 else 1         # first block in each stage has stride 2, else stride 1
                block = self.build_local_block(     # build local block: MBConv
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None) # wrap in ResidualBlock with IdentityLayer if not the first block
                stage.append(block)
                in_channels = w                     # output channel-depth of a block becomes input channel-depth for the next one.
            self.stages.append(OpSequential(stage)) # wrap in OpSequential
            self.width_list.append(in_channels)     # log the dimension of output channel-depth of block

        #################################################################################
        #                 Novel stages of EfficientViT (stages 3 and 4)                 #
        #                     MBConv + EfficientViT's novel module                      #
        #################################################################################

        # Loops over the rest of width and depth, i.e. stage 3 then stage 4
        # width_list=[-, -, -, 128, 256],
        # depth_list=[-, -, -, 3, 4],
        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(         # stage starts with ONE local_block
                in_channels=in_channels,
                out_channels=w,
                stride=2,                           # stride 2
                expand_ratio=expand_ratio,          # of type MBConv
                norm=norm,
                act_func=act_func,
                fewer_norm=True,                    # and with fewer_norm, affecting bias and norm
            )
            stage.append(ResidualBlock(block, None))# wrap in ResidualBlock, BUT no identity layer = no residual connection!
            in_channels = w                         # its output width is input to next

            for _ in range(d):                      # then depth[] number of the novel EViTBlocks!
                stage.append(                       # They are appended b2b on each other
                    QEfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage)) # append the structure to the model
            self.width_list.append(in_channels)     # log the dimension
        self.stages = nn.ModuleList(self.stages)    # Convert all stags to PyTorch ModuleList, creating a PyTorch model


    # Helper method that builds DSConv for the input stem and MBConv for the other stages
    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        """
        Static method to build a local block for the EfficientViTBackbone network.

        Parameters:
        in_channels (int): The number of input channels to the block.
        out_channels (int): The number of output channels from the block.
        stride (int): The stride of the convolution operation in the block.
        expand_ratio (float): The ratio by which to expand the channels in the block.
        norm (str): The type of normalization to use in the block.
        act_func (str): The activation function to use in the block.
        fewer_norm (bool, optional): A flag indicating whether to use fewer normalization layers. Defaults to False.

        Returns:
        nn.Module: The created block, which is either a DSConv block if expand_ratio is 1, or an MBConv block otherwise.
        """
        if expand_ratio == 1:
            block = QDSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = QMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    # function that forwards an input x through the backbone. Returns a dictionary including all intermediary feature maps 
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass through the EfficientViTBackbone.

        Parameters:
        x (torch.Tensor): The input tensor to the network.

        Returns:
        dict[str, torch.Tensor]: A dictionary containing the output tensors from each stage of the network, 
        as well as the final output tensor. The keys of the dictionary are the names of the stages 
        ("input", "stage0", "stage1", ..., "stage_final"), and the values are the output tensors from 
        each stage.
        """
        output_dict = {"input": x}                              # the dictionary keys are getting built on the fly
        output_dict["stage0"] = x = self.input_stem(x)          # Process through input stem
        for stage_id, stage in enumerate(self.stages, 1):
            output_dict["stage%d" % stage_id] = x = stage(x)    # Process through each stage
        output_dict["stage_final"] = x
        return output_dict                                      # All intermediate feautre maps are available in the dictionary, so that several features can be merged in the neck later
    

def efficientvit_backbone_b1_quant(**kwargs) -> EfficientViTBackboneQuant:
    # Step 4: Specify details of quantized backbone, and call for it using EfficientViTBackboneQuant
    backbone = EfficientViTBackboneQuant(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[1, 2, 3, 3, 4],
        dim=16,
        # filter the kwargs dict to match what EfficientVitBackboneQuant expects, and unpack the result with **
        **build_kwargs_from_config(kwargs, EfficientViTBackboneQuant),
    )
    return backbone

######################################################################
#                            Quantized SAM                           #
######################################################################

class EfficientViTLargeBackboneQuant(nn.Module):
    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        block_list: list[str] or None = None,
        expand_list: list[float] or None = None,
        fewer_norm_list: list[bool] or None = None,
        in_channels=3,                                  #RGB
        qkv_dim=32,
        norm="bn2d",                                    # Default normalization type
        act_func="gelu",                                # default activation type
        config=None,                                    # config gets passed as an (unpacked?) keywordargument
    ) -> None:
        super().__init__()

        self.config=config
        # configure block types
        block_list = block_list or ["res", "fmb", "fmb", "mb", "att"] # overridden in XL-models to ["res", "fmb", "fmb", "fmb", "att@3", "att@3"]
        # controls expansion in convolutions. 1 -> DSConv
        expand_list = expand_list or [1, 4, 4, 4, 6]                  # overriden in XL-models to [1, 4, 4, 4, 4, 6]
        fewer_norm_list = fewer_norm_list or [False, False, False, True, True] # overriden in XL-models to [False, False, False, False, True, True]
                                                                               # so always true for the last two stages
        self.width_list = []
        self.stages = []
        
        #################################################################################
        #                       Input Stem of EfficientViT-SAM                          #
        #          One Convolution + One or zero ResBlocks (model XL0 has zero)         #
        #################################################################################

        # First Convolution - Now always unquantized!
        stage0 = [
            QConvLayer(
                in_channels=3,              # RGB input
                out_channels=width_list[0], # width_list=[16, _, _, _, _] --> 16 kernels
                stride=2,                   # Stride 2
                norm=norm,
                act_func=act_func,
                 # configs
                config = self.config,
                stage_id="stage0",
                block_position=0,            # block position within the stage
                layer_position=0,
                block_name='independent',    # the first layer is not part of any block
                block_is_bottleneck=True,
            )
        ]

        # "ResBlocks" from ResNet34 - specifically depth_list[0] number of ResBlocks
        for i in range(depth_list[0]):
            block = self.build_local_block(
                block=block_list[0],            # always "res" = ResBlock
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=expand_list[0],    # Does not cause DSConv, as in Classification
                norm=norm,                      # None, b2nd
                act_func=act_func,              # gelu, None
                fewer_norm=fewer_norm_list[0],  # False
                # configs
                config = self.config,
                stage_id="stage0",
                block_position=i+1,              # block position within the stage
                block_name=block_list[0],        # res, passed to the QConvLayer
                block_is_bottleneck=False,
            )
            stage0.append(ResidualBlock(block, IdentityLayer())) # residual connection for each block
        # save the channel depth at output of the stem
        in_channels = width_list[0]
        self.stages.append(OpSequential(stage0))
        self.width_list.append(in_channels)

        #################################################################################
        #                         All other stages of EfficientViT-SAM                  #
        #        (MBConv OR Fused-MBConv) + (Attention OR MBconv OR Fused-MBConv)       #
        #################################################################################

        # A stage = a sequence of blocks.
        # A block = a build_local_block NOT wrapped in a ResidualBlock.

        # for all stages except the input stage
        for stage_id, (w, d) in enumerate(zip(width_list[1:], depth_list[1:]), start=1):
            stage = []
            # Start with ONE MBConb or Fused-MBConv, nothing else allowed
            block = self.build_local_block(
                block="mb" if block_list[stage_id] not in ["mb", "fmb"] else block_list[stage_id],
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_list[stage_id] * 4, # The first block in each of these stages have 4x larger expansion
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[stage_id],   # only true in the last two stages
                # config
                config = self.config,
                stage_id="stage%d" % stage_id,
                block_position=0,
                block_name="mb" if block_list[stage_id] not in ["mb", "fmb"] else block_list[stage_id],
                block_is_bottleneck=True,
            )
            stage.append(ResidualBlock(block, None)) #NO RESIDUAL CONNECTION for the first block
            in_channels = w

            # Build many more blocks, either attention or convolutions
            for i in range(d):
                if block_list[stage_id].startswith("att"):
                    stage.append(
                        QEfficientViTBlock(
                            in_channels=in_channels,
                            dim=qkv_dim,
                            expand_ratio=expand_list[stage_id],
                            scales=(3,) if block_list[stage_id] == "att@3" else (5,), # XL-models ONLY use 3scaling, all other only use 5-scaling
                            norm=norm,
                            act_func=act_func,
                            # config
                            config = self.config,
                            stage_id="stage%d" % stage_id,
                            block_position=i+1,
                            block_name=block_list[stage_id],
                            block_is_bottleneck=False,
                        )
                    )
                else: # build MBConv or FMBConv
                    block = self.build_local_block(
                        block=block_list[stage_id],
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=1,
                        expand_ratio=expand_list[stage_id],
                        norm=norm,
                        act_func=act_func,
                        fewer_norm=fewer_norm_list[stage_id],      # only true in the last two stages
                        # config
                        config = self.config,
                        stage_id="stage%d" % stage_id,
                        block_position=i+1,
                        block_name=block_list[stage_id],
                        block_is_bottleneck=False,
                    )
                    block = ResidualBlock(block, IdentityLayer()) # Residual connection for each other block
                    stage.append(block)
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    #################################################################################
    #                               Local block builder                             #
    #################################################################################
    @staticmethod
    def build_local_block(
        block: str,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
        **kwargs, # config arguments
    ) -> nn.Module:
        if block == "res":
            block = QResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,    # only true in the last two stages
                norm=(None, norm) if fewer_norm else norm,          # only true in the last two stages - else norm gets converted to a tuple of lenght 2
                act_func=(act_func, None),                          # blocks always end without an activation function
                **kwargs, # config arguments
            )
        elif block == "fmb":
            block = QFusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,    # only true in the last two stages
                norm=(None, norm) if fewer_norm else norm,          # only true in the last two stages - else norm gets converted to a tuple of lenght 2
                act_func=(act_func, None),                          # blocks always end without an activation function
                **kwargs, # config arguments
            )
        elif block == "mb":
            block = QMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False, # only true in the last two stages
                norm=(None, None, norm) if fewer_norm else norm,       # only true in the last two stages - else norm gets converted to a tuple of lenght 3
                act_func=(act_func, act_func, None),                   # blocks always end without an activation function
                **kwargs, # config arguments
            )
        else:
            raise ValueError(block)
        return block

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output_dict = {"input": x}
        for stage_id, stage in enumerate(self.stages):
            output_dict["stage%d" % stage_id] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict


def efficientvit_backbone_l0_quant(**kwargs) -> EfficientViTLargeBackboneQuant:
    backbone = EfficientViTLargeBackboneQuant(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 4, 4],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackboneQuant),
    )
    return backbone


def efficientvit_backbone_l1_quant(**kwargs) -> EfficientViTLargeBackboneQuant:
    backbone = EfficientViTLargeBackboneQuant(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 6, 6],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackboneQuant),
    )
    return backbone


def efficientvit_backbone_l2_quant(**kwargs) -> EfficientViTLargeBackboneQuant:
    backbone = EfficientViTLargeBackboneQuant(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 2, 2, 8, 8],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackboneQuant),
    )
    return backbone

# never used as no l3 model exist
def efficientvit_backbone_l3_quant(**kwargs) -> EfficientViTLargeBackboneQuant:
    backbone = EfficientViTLargeBackboneQuant(
        width_list=[64, 128, 256, 512, 1024],
        depth_list=[1, 2, 2, 8, 8],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackboneQuant),
    )
    return backbone
