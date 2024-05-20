# Backbone architectures for quantization experimentation
#for key in REGISTERED_BACKBONE_VERSIONS.keys(): print(key)


'''
This code creates and exports architectures for quantization of the image encoder, along with descriptions of each layer

The naming scheme is model : stage : block : layer.

When doing quantization, a key on this form will identfy a configuration from the dictionary REGISTERED_BACKBONE_VERSIONS.
This config is used by the function toggle_selective_attribute in the image encoder of sam.py
It is used to identify the layer which to toggle on.

Example:
L1:stage4:3:2 is the key that defines
Model L1
Stage 4 within the model
Block number 3 in the stage
Layer number 2 in the block

To get a description of this layer, call REGISTERED_BACKBONE_DESCRIPTIONS_LARGE[4:3:2]
and it will in this case return the string 'Proj' since this layer is a projection convolution inside an attention module.

Yes, there is an abnormality. When calling the dictionary, use form '4:3:2' without mention of model, but make sure to call the correct description dictionary (LARGE or XL).
When fetching a backbone version, use full form 'L1:stage4:3:2'

To view all keys in the terminal, uncomment the print statement inside create_backbone_versions.

The unquantized baseline model, which will not affect any layer, is named model:none:none:none. The backbone that will quantize all layers is named model:all:all:all.

You can define your own combinations. The code will toggle the attributes of those layers in the intersection of stage:block:layer, which are three lists that may contain several elements.
'''


def create_backbone_baselines():
    # Initialize a backbone with two baselines: quantize nothing (none:none:none) or everything (all:all:all)
    backbone_dict = {}
    models = ['L0','L1','L2','XL0','XL1']
    for m in models:
        backbone_dict[f'{m}:none:none:none'] =  {
            'stages': [],
            'block_positions': [],
            'layer_positions': [],
            }
    for m in models:
        backbone_dict[f'{m}:all:all:all'] =  {
                'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
                'block_positions': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], # includes all, as the deepest stage has 15 blocks counted from 0.
                'layer_positions': [0,1,2,3,4,5,6,7,8,9], # includes all
            }
    return backbone_dict


def create_backbone_versions(baseline_dict: dict):
    backbone_dict = baseline_dict
    models = ['L0','L1','L2', 'XL0', 'XL1']

    stages_L0 = ["stage0", "stage1", "stage2", "stage3", "stage4", "neck"]
    block_depth_per_stage_L0 = [2,2,2,1+4,1+4,3+4+1] # including bottleneck blocks
    layer_depth_per_block_L0 = [
        [1,2], # stage0 has 1 input conv and 1 resblock, which has 2 Convs
        [2,2], # stage1 has 1 + 1 FusedMBConv, they have 2 convs each
        [2,2], # stage2 has 1 + 1 FusedMBConv, they have 2 convs each
        [3,3,3,3,3], # stage3 has 1 + 4 MBConv, they have 3 convs each
        [3,6,6,6,6], # stage4 has 1 MBConv (which has 3 convs)+ 4 efficentViT Modules, which have 5 conv layers + 1 attention layer each (counting the trailing MBconv which has 3).
        [1,1,1,2,2,2,2,1]  # the neck has 3 blocks of conv+upsample, one learnable layer each. It has 4 FusedMBconvs, and 1 output conv
    ]

    stages_L1 = ["stage0", "stage1", "stage2", "stage3", "stage4", "neck"]
    block_depth_per_stage_L1 = [2,2,2,1+6,1+6,3+8+1] # including bottleneck blocks
    layer_depth_per_block_L1 = [
        [1,2], # stage0 has 1 input conv and 1 resblock, which has 2 Convs
        [2,2], # stage1 has 1 + 1 FusedMBConv, they have 2 convs each
        [2,2], # stage2 has 1 + 1 FusedMBConv, they have 2 convs each
        [3,3,3,3,3,3,3], # stage3 has 1 + 4 MBConv, they have 3 convs each
        [3,6,6,6,6,6,6], # stage4 has 1 MBConv (which has 3 convs)+ 4 efficentViT Modules, which have 6 conv layers + 1 attention layer each (counting the trailing MBconv which has 3).
        [1,1,1,2,2,2,2,2,2,2,2,1]  # the neck has 3 blocks of conv+upsample, one learnable layer each. It has 8 FusedMBconvs, and 1 output conv
    ]
    
    stages_L2 = ["stage0", "stage1", "stage2", "stage3", "stage4", "neck"]
    block_depth_per_stage_L2 = [2,2,2,1+8,1+8,3+12+1] # including bottleneck blocks
    layer_depth_per_block_L2 = [
        [1,2], # stage0 has 1 input conv and 1 resblock, which has 2 Convs
        [2,2], # stage1 has 1 + 1 FusedMBConv, they have 2 convs each
        [2,2], # stage2 has 1 + 1 FusedMBConv, they have 2 convs each
        [3,3,3,3,3,3,3,3,3], # stage3 has 1 + 4 MBConv, they have 3 convs each
        [3,6,6,6,6,6,6,6,6], # stage4 has 1 MBConv (which has 3 convs)+ 4 efficentViT Modules, which have 6 conv layers + 1 attention layer each (counting the trailing MBconv which has 3).
        [1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,1]  # the neck has 3 blocks of conv+upsample, one learnable layer each. It has 12 FusedMBconvs, and 1 output conv
    ]

    stages_XL0 = ["stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"]
    block_depth_per_stage_XL0 = [1, 2, 2, 1+2, 1+3, 1+3, 3+6+1] # including bottleneck blocks
    layer_depth_per_block_XL0 = [
        [1,], # stage0 has 1 input, no res block
        [2,2], # stage1 has 1 + 1 FusedMBConv, they have 2 convs each
        [2,2], # stage2 has 1 + 1 FusedMBConv, they have 2 convs each
        [2,2,2], # stage3 has 1 + 2 FusedMBConv, they have 2 convs each
        [3,6,6,6], # stage4 has 1 MBConv (which has 3 convs)+ 3 efficentViT Modules, which have 6 conv layers + 1 attention layer each (counting the trailing MBconv which has 3).
        [3,6,6,6], # stage5 is the same
        [1,1,1,2,2,2,2,2,2,1]  # the neck has 3 blocks of conv+upsample, one learnable layer each. It has 6 FusedMBconvs, and 1 output conv
    ]

    
    stages_XL1 = ["stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"]
    block_depth_per_stage_XL1 = [2, 1+4, 1+4, 1+4, 1+4, 1+6, 3+12+1] # including bottleneck blocks
    layer_depth_per_block_XL1 = [
        [1,2], # stage0 has 1 input, 1 resblock
        [2,2,2,2,2], # stage1 has 1 + 4 FusedMBConv, they have 2 convs each
        [2,2,2,2,2], # stage2 has 1 + 4 FusedMBConv, they have 2 convs each
        [2,2,2,2,2], # stage3 has 1 + 4 FusedMBConv, they have 2 convs each
        [3,6,6,6,6], # stage4 has 1 MBConv (which has 3 convs)+ 4 efficentViT Modules, which have 6 conv layers + 1 attention layer each (counting the trailing MBconv which has 3).
        [3,6,6,6,6,6,6], # stage5 is the same, but with 6 attention modules
        [1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,1]  # the neck has 3 blocks of conv+upsample, one learnable layer each. It has 6 FusedMBconvs, and 1 output conv
    ]

    for m in models:
        # layer granularity
        for s, block_depth, layer_depth in zip(eval(f'stages_{m}'), eval(f'block_depth_per_stage_{m}'), eval(f'layer_depth_per_block_{m}')):
            for bd in range(block_depth):
                for ld in range(layer_depth[bd]):
                    key = f"{m}:{s}:{bd}:{ld}"
                    value = {
                        'stages': [s],
                        'block_positions': [bd],
                        'layer_positions': [ld],
                    }
                    backbone_dict[key] = value

        # Uncomment below if you need the keys for scripting, or to print all
        #for key in backbone_dict.keys():
            #print(key)
            #print(key, backbone_dict[key])

    return backbone_dict

REGISTERED_BACKBONE_VERSIONS = create_backbone_versions(create_backbone_baselines())

REGISTERED_BACKBONE_DESCRIPTIONS_LARGE = {
    # Stage 0 - common to all large models
    '0:0:0': 'Conv',
    '0:1:0': 'ResBlock',
    '0:1:1': 'Resblock',
    
    # Stage 1 - common to all large models
    '1:0:0': 'FMBC 3x3',
    '1:0:1': 'FMBC 1x1',
    '1:1:0': 'FMBC 3x3',
    '1:1:1': 'FMBC 1x1',
    
    # Stage 2 - common to all large models
    '2:0:0': 'FMBC 3x3',
    '2:0:1': 'FMBC 1x1',
    '2:1:0': 'FMBC 3x3',
    '2:1:1': 'FMBC 1x1',
    
    # Stage 3
    '3:0:0': 'MBC 1x1 exp',
    '3:0:1': 'MBC DW',
    '3:0:2': 'MBC 1x1 comp',
    '3:1:0': 'MBC 1x1 exp',
    '3:1:1': 'MBC DW',
    '3:1:2': 'MBC 1x1 comp',
    '3:2:0': 'MBC 1x1 exp',
    '3:2:1': 'MBC DW',
    '3:2:2': 'MBC 1x1 comp',
    '3:3:0': 'MBC 1x1 exp',
    '3:3:1': 'MBC DW',
    '3:3:2': 'MBC 1x1 comp',
    '3:4:0': 'MBC 1x1 exp',
    '3:4:1': 'MBC DW',
    '3:4:2': 'MBC 1x1 comp',
    # L2
    '3:5:0': 'MBC 1x1 exp',
    '3:5:1': 'MBC DW',
    '3:5:2': 'MBC 1x1 comp',
    '3:6:0': 'MBC 1x1 exp',
    '3:6:1': 'MBC DW',
    '3:6:2': 'MBC 1x1 comp',
    # L3
    '3:7:0': 'MBC 1x1 exp',
    '3:7:1': 'MBC DW',
    '3:7:2': 'MBC 1x1 comp',
    '3:8:0': 'MBC 1x1 exp',
    '3:8:1': 'MBC DW',
    '3:8:2': 'MBC 1x1 comp',

    # Stage 4
    '4:0:0': 'MBC 1x1 exp',
    '4:0:1': 'MBC DW',
    '4:0:2': 'MBC 1x1 comp',
    '4:1:0': 'QKV',
    '4:1:1': 'Attention',
    '4:1:2': 'Proj',
    '4:1:3': 'MBC 1x1 exp',
    '4:1:4': 'MBC DW',
    '4:1:5': 'MBC 1x1 comp',
    '4:2:0': 'QKV',
    '4:2:1': 'Attention',
    '4:2:2': 'Proj',
    '4:2:3': 'MBC 1x1 exp',
    '4:2:4': 'MBC DW',
    '4:2:5': 'MBC 1x1 comp',
    '4:3:0': 'QKV',
    '4:3:1': 'Attention',
    '4:3:2': 'Proj',
    '4:3:3': 'MBC 1x1 exp',
    '4:3:4': 'MBC DW',
    '4:3:5': 'MBC 1x1 comp',
    '4:4:0': 'QKV',
    '4:4:1': 'Attention',
    '4:4:2': 'Proj',
    '4:4:3': 'MBC 1x1 exp',
    '4:4:4': 'MBC DW',
    '4:4:5': 'MBC 1x1 comp',

    # L2 - two extra attention modules
    '4:5:0': 'QKV',
    '4:5:1': 'Attention',
    '4:5:2': 'Proj',
    '4:5:3': 'MBC 1x1 exp',
    '4:5:4': 'MBC DW',
    '4:5:5': 'MBC 1x1 comp',
    '4:6:0': 'QKV',
    '4:6:1': 'Attention',
    '4:6:2': 'Proj',
    '4:6:3': 'MBC 1x1 exp',
    '4:6:4': 'MBC DW',
    '4:6:5': 'MBC 1x1 comp',    
  
    # L3 - two extra attention modules
    '4:7:0': 'QKV',
    '4:7:1': 'Attention',
    '4:7:2': 'Proj',
    '4:7:3': 'MBC 1x1 exp',
    '4:7:4': 'MBC DW',
    '4:7:5': 'MBC 1x1 comp',
    '4:8:0': 'QKV',
    '4:8:1': 'Attention',
    '4:8:2': 'Proj',
    '4:8:3': 'MBC 1x1 exp',
    '4:8:4': 'MBC DW',
    '4:8:5': 'MBC 1x1 comp',

    # Stage 5
    'neck:0:0': 'Conv + upsample from 512',
    'neck:1:0': 'Conv + upsample from 256',
    'neck:2:0': 'Conv + upsample from 128',
    'neck:3:0': 'FMBC 3x3',
    'neck:3:1': 'FMBC 1x1',
    'neck:4:0': 'FMBC 3x3',
    'neck:4:1': 'FMBC 1x1',
    'neck:5:0': 'FMBC 3x3',
    'neck:5:1': 'FMBC 1x1',
    'neck:6:0': 'FMBC 3x3',
    'neck:6:1': 'FMBC 1x1',

    # L2
    'neck:7:0': 'FMBC 3x3 OR Conv 1x1 in L0',
    'neck:7:1': 'FMBC 1x1',
    'neck:8:0': 'FMBC 3x3',
    'neck:8:1': 'FMBC 1x1',
    'neck:9:0': 'FMBC 3x3',
    'neck:9:1': 'FMBC 1x1',
    'neck:10:0': 'FMBC 3x3',
    'neck:10:1': 'FMBC 1x1',

    # L3 
    'neck:11:0': 'FMBC 3x3  OR Conv 1x1 in L1',
    'neck:11:1': 'FMBC 1x1',
    'neck:12:0': 'FMBC 3x3',
    'neck:12:1': 'FMBC 1x1',
    'neck:13:0': 'FMBC 3x3',
    'neck:13:1': 'FMBC 1x1',
    'neck:14:0': 'FMBC 3x3',
    'neck:14:1': 'FMBC 1x1',
    # common last layer
    'neck:15:0': 'Conv 1x1,',
    }

REGISTERED_BACKBONE_DESCRIPTIONS_XL = {
    # Stage 0
    '0:0:0': 'Conv',
    # XL1
    '0:1:0': 'ResBlock',
    '0:1:1': 'Resblock',
    
    # Stage 1
    '1:0:0': 'FMBC 3x3',
    '1:0:1': 'FMBC 1x1',
    '1:1:0': 'FMBC 3x3',
    '1:1:1': 'FMBC 1x1',
    # XL1 - three extra FMBC
    '1:2:0': 'FMBC 3x3',
    '1:2:1': 'FMBC 1x1',
    '1:3:0': 'FMBC 3x3',
    '1:3:1': 'FMBC 1x1',
    '1:4:0': 'FMBC 3x3',
    '1:4:1': 'FMBC 1x1',
    
    # Stage 2
    '2:0:0': 'FMBC 3x3',
    '2:0:1': 'FMBC 1x1',
    '2:1:0': 'FMBC 3x3',
    '2:1:1': 'FMBC 1x1',
    # XL1 - three extra FMBC
    '2:2:0': 'FMBC 3x3',
    '2:2:1': 'FMBC 1x1',
    '2:3:0': 'FMBC 3x3',
    '2:3:1': 'FMBC 1x1',
    '2:4:0': 'FMBC 3x3',
    '2:4:1': 'FMBC 1x1',
    
    # Stage 3
    '3:0:0': 'FMBC 3x3',
    '3:0:1': 'FMBC 1x1',
    '3:1:0': 'FMBC 3x3',
    '3:1:1': 'FMBC 1x1',
    '3:2:0': 'FMBC 3x3',
    '3:2:1': 'FMBC 1x1',
    # XL1 - two extra FMBC
    '3:3:0': 'FMBC 3x3',
    '3:3:1': 'FMBC 1x1',
    '3:4:0': 'FMBC 3x3',
    '3:4:1': 'FMBC 1x1',

    # Stage 4
    '4:0:0': 'MBC 1x1 exp',
    '4:0:1': 'MBC DW',
    '4:0:2': 'MBC 1x1 comp',
    '4:1:0': 'QKV',
    '4:1:1': 'Attention',
    '4:1:2': 'Proj',
    '4:1:3': 'MBC 1x1 exp',
    '4:1:4': 'MBC DW',
    '4:1:5': 'MBC 1x1 comp',
    '4:2:0': 'QKV',
    '4:2:1': 'Attention',
    '4:2:2': 'Proj',
    '4:2:3': 'MBC 1x1 exp',
    '4:2:4': 'MBC DW',
    '4:2:5': 'MBC 1x1 comp',
    '4:3:0': 'QKV',
    '4:3:1': 'Attention',
    '4:3:2': 'Proj',
    '4:3:3': 'MBC 1x1 exp',
    '4:3:4': 'MBC DW',
    '4:3:5': 'MBC 1x1 comp',
    # XL1 - one extra attention module
    '4:4:0': 'QKV',
    '4:4:1': 'Attention',
    '4:4:2': 'Proj',
    '4:4:3': 'MBC 1x1 exp',
    '4:4:4': 'MBC DW',
    '4:4:5': 'MBC 1x1 comp',

    # Stage 5
    '5:0:0': 'MBC 1x1 exp',
    '5:0:1': 'MBC DW',
    '5:0:2': 'MBC 1x1 comp',
    '5:1:0': 'QKV',
    '5:1:1': 'Attention',
    '5:1:2': 'Proj',
    '5:1:3': 'MBC 1x1 exp',
    '5:1:4': 'MBC DW',
    '5:1:5': 'MBC 1x1 comp',
    '5:2:0': 'QKV',
    '5:2:1': 'Attention',
    '5:2:2': 'Proj',
    '5:2:3': 'MBC 1x1 exp',
    '5:2:4': 'MBC DW',
    '5:2:5': 'MBC 1x1 comp',
    '5:3:0': 'QKV',
    '5:3:1': 'Attention',
    '5:3:2': 'Proj',
    '5:3:3': 'MBC 1x1 exp',
    '5:3:4': 'MBC DW',
    '5:3:5': 'MBC 1x1 comp',
    # XL1 - three extra attention module
    '5:4:0': 'QKV',
    '5:4:1': 'Attention',
    '5:4:2': 'Proj',
    '5:4:3': 'MBC 1x1 exp',
    '5:4:4': 'MBC DW',
    '5:4:5': 'MBC 1x1 comp',
    '5:5:0': 'QKV',
    '5:5:1': 'Attention',
    '5:5:2': 'Proj',
    '5:5:3': 'MBC 1x1 exp',
    '5:5:4': 'MBC DW',
    '5:5:5': 'MBC 1x1 comp',
    '5:6:0': 'QKV',
    '5:6:1': 'Attention',
    '5:6:2': 'Proj',
    '5:6:3': 'MBC 1x1 exp',
    '5:6:4': 'MBC DW',
    '5:6:5': 'MBC 1x1 comp',

    # neck
    'neck:0:0': 'Conv + upsample from 512',
    'neck:1:0': 'Conv + upsample from 256',
    'neck:2:0': 'Conv + upsample from 128',
    'neck:3:0': 'FMBC 3x3',
    'neck:3:1': 'FMBC 1x1',
    'neck:4:0': 'FMBC 3x3',
    'neck:4:1': 'FMBC 1x1',
    'neck:5:0': 'FMBC 3x3',
    'neck:5:1': 'FMBC 1x1',
    'neck:6:0': 'FMBC 3x3',
    'neck:6:1': 'FMBC 1x1',

    # XL1 has six extra FMBC
    'neck:7:0': 'FMBC 3x3 OR Conv 1x1 in XL0',
    'neck:7:1': 'FMBC 1x1',
    'neck:8:0': 'FMBC 3x3',
    'neck:8:1': 'FMBC 1x1',
    'neck:9:0': 'FMBC 3x3',
    'neck:9:1': 'FMBC 1x1',
    'neck:10:0': 'FMBC 3x3',
    'neck:10:1': 'FMBC 1x1',
    'neck:11:0': 'FMBC 3x3',
    'neck:11:1': 'FMBC 1x1',
    'neck:12:0': 'FMBC 3x3',
    'neck:12:1': 'FMBC 1x1',
    'neck:13:0': 'FMBC 3x3',
    'neck:13:1': 'FMBC 1x1',
    'neck:14:0': 'FMBC 3x3',
    'neck:14:1': 'FMBC 1x1',
    # common last layer, which is the only layer for which the naming scheme breaks.
    # XLO has neck:7:0 as last layer, XL1 has neck:15:0
    'neck:15:0': 'Conv 1x1,',
    }
