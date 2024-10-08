# Author: Joel Bengs. Apache-2.0 license
# This file is a library for the pre-defined backbone variants.
# They are defined in config/quant_backbone_zoo.py

backbones_baselines=(
L0:none:none:none
L1:none:none:none
L2:none:none:none
XL0:none:none:none
XL1:none:none:none
L0:all:all:all
L1:all:all:all
L2:all:all:all
XL0:all:all:all
XL1:all:all:all
)


backbones_L0=(
L0:stage0:0:0
L0:stage0:1:0
L0:stage0:1:1
L0:stage1:0:0
L0:stage1:0:1
L0:stage1:1:0
L0:stage1:1:1
L0:stage2:0:0
L0:stage2:0:1
L0:stage2:1:0
L0:stage2:1:1
L0:stage3:0:0
L0:stage3:0:1
L0:stage3:0:2
L0:stage3:1:0
L0:stage3:1:1
L0:stage3:1:2
L0:stage3:2:0
L0:stage3:2:1
L0:stage3:2:2
L0:stage3:3:0
L0:stage3:3:1
L0:stage3:3:2
L0:stage3:4:0
L0:stage3:4:1
L0:stage3:4:2
L0:stage4:0:0
L0:stage4:0:1
L0:stage4:0:2
L0:stage4:1:0
L0:stage4:1:1
L0:stage4:1:2
L0:stage4:1:3
L0:stage4:1:4
L0:stage4:1:5
L0:stage4:2:0
L0:stage4:2:1
L0:stage4:2:2
L0:stage4:2:3
L0:stage4:2:4
L0:stage4:2:5
L0:stage4:3:0
L0:stage4:3:1
L0:stage4:3:2
L0:stage4:3:3
L0:stage4:3:4
L0:stage4:3:5
L0:stage4:4:0
L0:stage4:4:1
L0:stage4:4:2
L0:stage4:4:3
L0:stage4:4:4
L0:stage4:4:5
L0:neck:0:0
L0:neck:1:0
L0:neck:2:0
L0:neck:3:0
L0:neck:3:1
L0:neck:4:0
L0:neck:4:1
L0:neck:5:0
L0:neck:5:1
L0:neck:6:0
L0:neck:6:1
L0:neck:7:0
)

backbones_L1=(
L1:stage0:0:0
L1:stage0:1:0
L1:stage0:1:1
L1:stage1:0:0
L1:stage1:0:1
L1:stage1:1:0
L1:stage1:1:1
L1:stage2:0:0
L1:stage2:0:1
L1:stage2:1:0
L1:stage2:1:1
L1:stage3:0:0
L1:stage3:0:1
L1:stage3:0:2
L1:stage3:1:0
L1:stage3:1:1
L1:stage3:1:2
L1:stage3:2:0
L1:stage3:2:1
L1:stage3:2:2
L1:stage3:3:0
L1:stage3:3:1
L1:stage3:3:2
L1:stage3:4:0
L1:stage3:4:1
L1:stage3:4:2
L1:stage3:5:0
L1:stage3:5:1
L1:stage3:5:2
L1:stage3:6:0
L1:stage3:6:1
L1:stage3:6:2
L1:stage4:0:0
L1:stage4:0:1
L1:stage4:0:2
L1:stage4:1:0
L1:stage4:1:1
L1:stage4:1:2
L1:stage4:1:3
L1:stage4:1:4
L1:stage4:1:5
L1:stage4:2:0
L1:stage4:2:1
L1:stage4:2:2
L1:stage4:2:3
L1:stage4:2:4
L1:stage4:2:5
L1:stage4:3:0
L1:stage4:3:1
L1:stage4:3:2
L1:stage4:3:3
L1:stage4:3:4
L1:stage4:3:5
L1:stage4:4:0
L1:stage4:4:1
L1:stage4:4:2
L1:stage4:4:3
L1:stage4:4:4
L1:stage4:4:5
L1:stage4:5:0
L1:stage4:5:1
L1:stage4:5:2
L1:stage4:5:3
L1:stage4:5:4
L1:stage4:5:5
L1:stage4:6:0
L1:stage4:6:1
L1:stage4:6:2
L1:stage4:6:3
L1:stage4:6:4
L1:stage4:6:5
L1:neck:0:0
L1:neck:1:0
L1:neck:2:0
L1:neck:3:0
L1:neck:3:1
L1:neck:4:0
L1:neck:4:1
L1:neck:5:0
L1:neck:5:1
L1:neck:6:0
L1:neck:6:1
L1:neck:7:0
L1:neck:7:1
L1:neck:8:0
L1:neck:8:1
L1:neck:9:0
L1:neck:9:1
L1:neck:10:0
L1:neck:10:1
L1:neck:11:0
)

backbones_L2=(
L2:stage0:0:0
L2:stage0:1:0
L2:stage0:1:1
L2:stage1:0:0
L2:stage1:0:1
L2:stage1:1:0
L2:stage1:1:1
L2:stage2:0:0
L2:stage2:0:1
L2:stage2:1:0
L2:stage2:1:1
L2:stage3:0:0
L2:stage3:0:1
L2:stage3:0:2
L2:stage3:1:0
L2:stage3:1:1
L2:stage3:1:2
L2:stage3:2:0
L2:stage3:2:1
L2:stage3:2:2
L2:stage3:3:0
L2:stage3:3:1
L2:stage3:3:2
L2:stage3:4:0
L2:stage3:4:1
L2:stage3:4:2
L2:stage3:5:0
L2:stage3:5:1
L2:stage3:5:2
L2:stage3:6:0
L2:stage3:6:1
L2:stage3:6:2
L2:stage3:7:0
L2:stage3:7:1
L2:stage3:7:2
L2:stage3:8:0
L2:stage3:8:1
L2:stage3:8:2
L2:stage4:0:0
L2:stage4:0:1
L2:stage4:0:2
L2:stage4:1:0
L2:stage4:1:1
L2:stage4:1:2
L2:stage4:1:3
L2:stage4:1:4
L2:stage4:1:5
L2:stage4:2:0
L2:stage4:2:1
L2:stage4:2:2
L2:stage4:2:3
L2:stage4:2:4
L2:stage4:2:5
L2:stage4:3:0
L2:stage4:3:1
L2:stage4:3:2
L2:stage4:3:3
L2:stage4:3:4
L2:stage4:3:5
L2:stage4:4:0
L2:stage4:4:1
L2:stage4:4:2
L2:stage4:4:3
L2:stage4:4:4
L2:stage4:4:5
L2:stage4:5:0
L2:stage4:5:1
L2:stage4:5:2
L2:stage4:5:3
L2:stage4:5:4
L2:stage4:5:5
L2:stage4:6:0
L2:stage4:6:1
L2:stage4:6:2
L2:stage4:6:3
L2:stage4:6:4
L2:stage4:6:5
L2:stage4:7:0
L2:stage4:7:1
L2:stage4:7:2
L2:stage4:7:3
L2:stage4:7:4
L2:stage4:7:5
L2:stage4:8:0
L2:stage4:8:1
L2:stage4:8:2
L2:stage4:8:3
L2:stage4:8:4
L2:stage4:8:5
L2:neck:0:0
L2:neck:1:0
L2:neck:2:0
L2:neck:3:0
L2:neck:3:1
L2:neck:4:0
L2:neck:4:1
L2:neck:5:0
L2:neck:5:1
L2:neck:6:0
L2:neck:6:1
L2:neck:7:0
L2:neck:7:1
L2:neck:8:0
L2:neck:8:1
L2:neck:9:0
L2:neck:9:1
L2:neck:10:0
L2:neck:10:1
L2:neck:11:0
L2:neck:11:1
L2:neck:12:0
L2:neck:12:1
L2:neck:13:0
L2:neck:13:1
L2:neck:14:0
L2:neck:14:1
L2:neck:15:0
)

backbones_XL0=(
XL0:stage0:0:0
XL0:stage1:0:0
XL0:stage1:0:1
XL0:stage1:1:0
XL0:stage1:1:1
XL0:stage2:0:0
XL0:stage2:0:1
XL0:stage2:1:0
XL0:stage2:1:1
XL0:stage3:0:0
XL0:stage3:0:1
XL0:stage3:1:0
XL0:stage3:1:1
XL0:stage3:2:0
XL0:stage3:2:1
XL0:stage4:0:0
XL0:stage4:0:1
XL0:stage4:0:2
XL0:stage4:1:0
XL0:stage4:1:1
XL0:stage4:1:2
XL0:stage4:1:3
XL0:stage4:1:4
XL0:stage4:1:5
XL0:stage4:2:0
XL0:stage4:2:1
XL0:stage4:2:2
XL0:stage4:2:3
XL0:stage4:2:4
XL0:stage4:2:5
XL0:stage4:3:0
XL0:stage4:3:1
XL0:stage4:3:2
XL0:stage4:3:3
XL0:stage4:3:4
XL0:stage4:3:5
XL0:stage5:0:0
XL0:stage5:0:1
XL0:stage5:0:2
XL0:stage5:1:0
XL0:stage5:1:1
XL0:stage5:1:2
XL0:stage5:1:3
XL0:stage5:1:4
XL0:stage5:1:5
XL0:stage5:2:0
XL0:stage5:2:1
XL0:stage5:2:2
XL0:stage5:2:3
XL0:stage5:2:4
XL0:stage5:2:5
XL0:stage5:3:0
XL0:stage5:3:1
XL0:stage5:3:2
XL0:stage5:3:3
XL0:stage5:3:4
XL0:stage5:3:5
XL0:neck:0:0
XL0:neck:1:0
XL0:neck:2:0
XL0:neck:3:0
XL0:neck:3:1
XL0:neck:4:0
XL0:neck:4:1
XL0:neck:5:0
XL0:neck:5:1
XL0:neck:6:0
XL0:neck:6:1
XL0:neck:7:0
XL0:neck:7:1
XL0:neck:8:0
XL0:neck:8:1
XL0:neck:9:0
)

backbones_XL1=(
XL1:stage0:0:0
XL1:stage0:1:0
XL1:stage0:1:1
XL1:stage1:0:0
XL1:stage1:0:1
XL1:stage1:1:0
XL1:stage1:1:1
XL1:stage1:2:0
XL1:stage1:2:1
XL1:stage1:3:0
XL1:stage1:3:1
XL1:stage1:4:0
XL1:stage1:4:1
XL1:stage2:0:0
XL1:stage2:0:1
XL1:stage2:1:0
XL1:stage2:1:1
XL1:stage2:2:0
XL1:stage2:2:1
XL1:stage2:3:0
XL1:stage2:3:1
XL1:stage2:4:0
XL1:stage2:4:1
XL1:stage3:0:0
XL1:stage3:0:1
XL1:stage3:1:0
XL1:stage3:1:1
XL1:stage3:2:0
XL1:stage3:2:1
XL1:stage3:3:0
XL1:stage3:3:1
XL1:stage3:4:0
XL1:stage3:4:1
XL1:stage4:0:0
XL1:stage4:0:1
XL1:stage4:0:2
XL1:stage4:1:0
XL1:stage4:1:1
XL1:stage4:1:2
XL1:stage4:1:3
XL1:stage4:1:4
XL1:stage4:1:5
XL1:stage4:2:0
XL1:stage4:2:1
XL1:stage4:2:2
XL1:stage4:2:3
XL1:stage4:2:4
XL1:stage4:2:5
XL1:stage4:3:0
XL1:stage4:3:1
XL1:stage4:3:2
XL1:stage4:3:3
XL1:stage4:3:4
XL1:stage4:3:5
XL1:stage4:4:0
XL1:stage4:4:1
XL1:stage4:4:2
XL1:stage4:4:3
XL1:stage4:4:4
XL1:stage4:4:5
XL1:stage5:0:0
XL1:stage5:0:1
XL1:stage5:0:2
XL1:stage5:1:0
XL1:stage5:1:1
XL1:stage5:1:2
XL1:stage5:1:3
XL1:stage5:1:4
XL1:stage5:1:5
XL1:stage5:2:0
XL1:stage5:2:1
XL1:stage5:2:2
XL1:stage5:2:3
XL1:stage5:2:4
XL1:stage5:2:5
XL1:stage5:3:0
XL1:stage5:3:1
XL1:stage5:3:2
XL1:stage5:3:3
XL1:stage5:3:4
XL1:stage5:3:5
XL1:stage5:4:0
XL1:stage5:4:1
XL1:stage5:4:2
XL1:stage5:4:3
XL1:stage5:4:4
XL1:stage5:4:5
XL1:stage5:5:0
XL1:stage5:5:1
XL1:stage5:5:2
XL1:stage5:5:3
XL1:stage5:5:4
XL1:stage5:5:5
XL1:stage5:6:0
XL1:stage5:6:1
XL1:stage5:6:2
XL1:stage5:6:3
XL1:stage5:6:4
XL1:stage5:6:5
XL1:neck:0:0
XL1:neck:1:0
XL1:neck:2:0
XL1:neck:3:0
XL1:neck:3:1
XL1:neck:4:0
XL1:neck:4:1
XL1:neck:5:0
XL1:neck:5:1
XL1:neck:6:0
XL1:neck:6:1
XL1:neck:7:0
XL1:neck:7:1
XL1:neck:8:0
XL1:neck:8:1
XL1:neck:9:0
XL1:neck:9:1
XL1:neck:10:0
XL1:neck:10:1
XL1:neck:11:0
XL1:neck:11:1
XL1:neck:12:0
XL1:neck:12:1
XL1:neck:13:0
XL1:neck:13:1
XL1:neck:14:0
XL1:neck:14:1
XL1:neck:15:0
)