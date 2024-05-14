# Backbone architectures for quantization experimentation
#for key in REGISTERED_BACKBONE_VERSIONS.keys(): print(key)


# Creates backbones for layerwise investigation of L0


def create_backbone_baselines():
    # Initialize a backbone with two baselines: quantize nothing (none:none:none) or everything (all:all:all)
    backbone_dict = {}
    models = ['L0','L1','L2','XL0','XL1']
    for m in models:
        backbone_dict[f'{m}:none:none:none'] =  {
            'stages': [],
            'block_positions': [],
            'layer_positions': [],
            },
    for m in models:
        backbone_dict[f'{m}:all:all:all'] =  {
                'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
                'block_positions': [0,1,2,3,4,5,6,7,8,9], # surely includes all
                'layer_positions': [0,1,2,3,4,5,6,7,8,9], # surely includes all
            },
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

REGISTERED_BACKBONE_DESCRIPTIONS = {
    ### Just t''o provide types
    '0:0:0': 'Input Conv',
    '0:1:0': 'ResBlock conv 1',
    '0:1:1': 'Resblock conv 2',
    '1:0:0': 'FMBConv 3x3',
    '1:0:1': 'FMBConv 1x1',
    '1:1:0': 'FMBConv 3x3',
    '1:1:1': 'FMBConv 1x1',
    '2:0:0': 'FMBConv 3x3',
    '2:0:1': 'FMBConv 1x1',
    '2:1:0': 'FMBConv 3x3',
    '2:1:1': 'FMBConv 1x1',
    '3:0:0': 'MBConv 1x1 expand',
    '3:0:1': 'MBConv DWconv',
    '3:0:2': 'MBConv 1x1 compress',
    '3:1:0': 'MBConv 1x1 expand',
    '3:1:1': 'MBConv DWconv',
    '3:1:2': 'MBConv 1x1 compress',
    '3:2:0': 'MBConv 1x1 expand',
    '3:2:1': 'MBConv DWconv',
    '3:2:2': 'MBConv 1x1 compress',
    '3:3:0': 'MBConv 1x1 expand',
    '3:3:1': 'MBConv DWconv',
    '3:3:2': 'MBConv 1x1 compress',
    '3:4:0': 'MBConv 1x1 expand',
    '3:4:1': 'MBConv DWconv',
    '3:4:2': 'MBConv 1x1 compress',
    '4:0:0': 'MBConv 1x1 expand',
    '4:0:1': 'MBConv DWconv',
    '4:0:2': 'MBConv 1x1 no a compress',
    '4:1:0': 'QKV-Conv 1x1',
    '4:1:1': 'Scaling Conv 5x5',
    '4:1:2': 'Scaling Conv 1x1',
    '4:1:3': 'Projection conv 1x1',
    '4:1:4': 'MBConv 1x1 expand',
    '4:1:5': 'MBConv DWconv',
    '4:1:6': 'MBConv 1x1 compress',
    '4:2:0': 'QKV-Conv 1x1',
    '4:2:1': 'Scaling Conv 5x5',
    '4:2:2': 'Scaling Conv 1x1',
    '4:2:3': 'Projection conv 1x1',
    '4:2:4': 'MBConv 1x1 expand',
    '4:2:5': 'MBConv DWconv',
    '4:2:6': 'MBConv 1x1 compress',
    '4:3:0': 'QKV-Conv 1x1',
    '4:3:1': 'Scaling Conv 5x5',
    '4:3:2': 'Scaling Conv 1x1',
    '4:3:3': 'Projection conv 1x1',
    '4:3:4': 'MBConv 1x1 expand',
    '4:3:5': 'MBConv DWconv',
    '4:3:6': 'MBConv 1x1 compress',
    '4:4:0': 'QKV-Conv 1x1',
    '4:4:1': 'Scaling Conv 5x5',
    '4:4:2': 'Scaling Conv 1x1',
    '4:4:3': 'Projection conv 1x1',
    '4:4:4': 'MBConv 1x1 expand',
    '4:4:5': 'MBConv DWconv',
    '4:4:6': 'MBConv 1x1 compress',
    'neck:0:0': 'Conv + upsample from 512',
    'neck:1:0': 'Conv + upsample from 256',
    'neck:2:0': 'Conv + upsample from 128',
    'neck:3:0': 'FMBConv 3x3',
    'neck:3:1': 'FMBConv 1x1',
    'neck:4:0': 'FMBConv 3x3',
    'neck:4:1': 'FMBConv 1x1',
    'neck:5:0': 'FMBConv 3x3',
    'neck:5:1': 'FMBConv 1x1',
    'neck:6:0': 'FMBConv 3x3',
    'neck:6:1': 'FMBConv 1x1',
    'neck:7:0': 'Conv 1x1,',
}

REGISTERED_BACKBONE_DESCRIPTIONS_DETAILED = {
    ### Just to provide types
    '0:0:0': 'Input Conv (bottleneck)',
    '0:1:0': 'ResBlock conv',
    '0:1:1': 'Resblock conv no act',
    '1:0:0': 'FMBConv 3x3 (bottleneck)',
    '1:0:1': 'FMBConv 1x1 no act (bottleneck)',
    '1:1:0': 'FMBConv 3x3',
    '1:1:1': 'FMBConv 1x1 no act',
    '2:0:0': 'FMBConv 3x3 (bottleneck)',
    '2:0:1': 'FMBConv 1x1 no act (bottleneck)',
    '2:1:0': 'FMBConv 3x3',
    '2:1:1': 'FMBConv 1x1 no act',
    '3:0:0': 'MBConv 1x1 no norm (bottleneck)',
    '3:0:1': 'MBConv DWconv no norm (bottleneck)',
    '3:0:2': 'MBConv 1x1 no act (bottleneck)',
    '3:1:0': 'MBConv 1x1 no norm',
    '3:1:1': 'MBConv DWconv no norm',
    '3:1:2': 'MBConv 1x1 no act',
    '3:2:0': 'MBConv 1x1 no norm',
    '3:2:1': 'MBConv DWconv no norm',
    '3:2:2': 'MBConv 1x1 no act',
    '3:3:0': 'MBConv 1x1 no norm',
    '3:3:1': 'MBConv DWconv no norm',
    '3:3:2': 'MBConv 1x1 no act',
    '3:4:0': 'MBConv 1x1 no norm',
    '3:4:1': 'MBConv DWconv no norm',
    '3:4:2': 'MBConv 1x1 no act',
    '4:0:0': 'MBConv 1x1 no norm (bottleneck)',
    '4:0:1': 'MBConv DWconv no norm (bottleneck)',
    '4:0:2': 'MBConv 1x1 no act (bottleneck)',
    '4:1:0': 'QKV-Conv 1x1',
    '4:1:1': 'Scaling Conv 5x5, no norm, no act',
    '4:1:2': 'Scaling Conv 1x1, no norm, no act',
    '4:1:3': 'Projection conv 1x1',
    '4:1:4': 'MBConv 1x1 no norm',
    '4:1:5': 'MBConv DWconv no norm',
    '4:1:6': 'MBConv 1x1 no act',
    '4:2:0': 'QKV-Conv 1x1',
    '4:2:1': 'Scaling Conv 5x5, no norm, no act',
    '4:2:2': 'Scaling Conv 1x1, no norm, no act',
    '4:2:3': 'Projection conv 1x1',
    '4:2:4': 'MBConv 1x1 no norm',
    '4:2:5': 'MBConv DWconv no norm',
    '4:2:6': 'MBConv 1x1 no act',
    '4:3:0': 'QKV-Conv 1x1',
    '4:3:1': 'Scaling Conv 5x5, no norm, no act',
    '4:3:2': 'Scaling Conv 1x1, no norm, no act',
    '4:3:3': 'Projection conv 1x1',
    '4:3:4': 'MBConv 1x1 no norm',
    '4:3:5': 'MBConv DWconv no norm',
    '4:3:6': 'MBConv 1x1 no act',
    '4:4:0': 'QKV-Conv 1x1',
    '4:4:1': 'Scaling Conv 5x5, no norm, no act',
    '4:4:2': 'Scaling Conv 1x1, no norm, no act',
    '4:4:3': 'Projection conv 1x1',
    '4:4:4': 'MBConv 1x1 no norm',
    '4:4:5': 'MBConv DWconv no norm',
    '4:4:6': 'MBConv 1x1 no act',
    'neck:0:0': 'Conv + upsample from 512, no act',
    'neck:1:0': 'Conv + upsample from 256, no act',
    'neck:2:0': 'Conv + upsample from 128, no act',
    'neck:3:0': 'FMBConv 3x3',
    'neck:3:1': 'FMBConv 1x1 no act',
    'neck:4:0': 'FMBConv 3x3',
    'neck:4:1': 'FMBConv 1x1 no act',
    'neck:5:0': 'FMBConv 3x3',
    'neck:5:1': 'FMBConv 1x1 no act',
    'neck:6:0': 'FMBConv 3x3',
    'neck:6:1': 'FMBConv 1x1 no act',
    'neck:7:0': 'Conv 1x1, no act',
}