# Backbone architectures for quantization experimentation
#for key in REGISTERED_BACKBONE_VERSIONS.keys(): print(key)

# Creates backbones for layerwise investigation of L0
def create_simple_backbone_versions_L0():
    stages_L0 = ["stage0", "stage1", "stage2", "stage3", "stage4", "neck"]
    block_depth_per_stage_L0 = [2,2,2,5,5,8] # including bottleneck blocks
    layer_depth_per_block_L0 = [
        [1,2], # stage0 has 1 input conv and 1 resblock, which has 2 Convs
        [2,2], # stage1 has 1 + 1 FusedMBConv, they have 2 convs each
        [2,2], # stage2 has 1 + 1 FusedMBConv, they have 2 convs each
        [3,3,3,3,3], # stage3 has 1 + 4 MBConv, they have 3 convs each
        [3,7,7,7,7], # stage4 has 1 MBConv (which has 3 convs)+ 4 efficentViT Modules, which have 67 conv layers each (counting the trailing MBconv which has 3). Note that multi-layer scaling has 2 layers!
        [1,1,1,2,2,2,2,1]  # the neck has 3 blocks of conv+upsample, one learnable layer each. It has 4 FusedMBconvs, and 1 output conv
    ]

    # Initialize a backbone with two baselines: quantize nothing (-:-:-) or everything (x:x:x)
    backbone_dict = {
        'L0:-:-:-': {
        'stages': [],
        'block_positions': [],
        'layer_positions': [],
        },
        'L0:x:x:x': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_positions': [0,1,2,3,4,5,6,7,8,9],
        'layer_positions': [0,1,2,3,4,5,6,7,8,9],
        }
     }

    # block granularity:
    for s, block_depth in zip(stages_L0, block_depth_per_stage_L0):
        for bd in range(block_depth):
            key = f"L0:{s}:{bd}:x"
            value = {
                'stages': [s],
                'block_positions': [bd]
            }
            backbone_dict[key] = value

    # layer granularity
    for s, block_depth, layer_depth in zip(stages_L0, block_depth_per_stage_L0, layer_depth_per_block_L0):
        for bd in range(block_depth):
            for ld in range(layer_depth[bd]):
                key = f"L0:{s}:{bd}:{ld}"
                value = {
                    'stages': [s],
                    'block_positions': [bd],
                    'layer_positions': [ld],
                }
                backbone_dict[key] = value

    # Uncomment below if you need the keys for scripting, or to see all
    #for key in backbone_dict.keys():
        #print(key)
        #print(key, backbone_dict[key])
    return backbone_dict

SIMPLE_REGISTERED_BACKBONE_VERSIONS = create_simple_backbone_versions_L0()

REGISTERED_BACKBONE_VERSIONS = {
    # BASELINES
    'FP32_baseline': {
        'stages': [],
        'block_names': [],
        'spare_bottlenecks': True,
        'spare_attention_qkv': True,
        'spare_attention_scaling': True,
        'spare_attention_projection': True,
    },
    'INT8_baseline': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },

    ### Family ###
    '3_q_all_but_stage0': {
        'stages': ["unknown", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '3_q_all_but_stage1': {
        'stages': ["unknown", "stage0", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '3_q_all_but_stage2': {
        'stages': ["unknown", "stage0", "stage1", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '3_q_all_but_stage3': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage4", "stage5", "neck"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '3_q_all_but_stage4': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage5", "neck"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '3_q_all_but_stage5': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "neck"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '3_q_all_but_neck': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,

    },

    ### Family ###
    '4_q_all_but_ResBlocks': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["independent", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '4_q_all_but_MBConvs': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["independent", "res", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '4_q_all_but_FusedMBConvs': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["independent", "res", "mb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '4_q_all_but_Attention': { #should give same results as quantizing all but saving stage 4 + 5 - do check that it is the case!
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["independent", "res", "mb", "fmb", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },

    ### Family ###
    '5_q_all_but_bottlenecks': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': True,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '5_q_all_but_attention_qkv': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': True,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '5_q_all_but_attention_scaling': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': True,
        'spare_attention_projection': False,
    },
    '5_q_all_but_attention_projection': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': True,
    },

    ### Family ###
    '6_q_only_stage0_spare_nothing': {
        'stages': ["stage0"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '6_q_only_stage1_spare_nothing': {
        'stages': ["stage1"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '6_q_only_stage2_spare_nothing': {
        'stages': ["stage2"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '6_q_only_stage3_spare_nothing': {
        'stages': ["stage3"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '6_q_only_stage4_spare_nothing': {
        'stages': ["stage4"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '6_q_only_stage5_spare_nothing': {
        'stages': ["stage5"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '6_q_only_neck_spare_nothing': {
        'stages': ["neck"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },

    ### Family ###
    '7_q_only_stage0_spare_bottlenecks': {
        'stages': ["stage0"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': True,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '7_q_only_stage1_spare_bottlenecks': {
        'stages': ["stage1"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': True,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '7_q_only_stage2_spare_bottlenecks': {
        'stages': ["stage2"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': True,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '7_q_only_stage3_spare_bottlenecks': {
        'stages': ["stage3"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': True,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '7_q_only_stage4_spare_bottlenecks': {
        'stages': ["stage4"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': True,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '7_q_only_stage5_spare_bottlenecks': {
        'stages': ["stage5"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': True,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '7_q_only_neck_spare_bottlenecks': {
        'stages': ["neck"],
        'block_names': ["independent", "res", "mb", "fmb", "att", "att@3", "att@5", "dag"],
        'spare_bottlenecks': True,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },

    ### Family ###
    '8_q_only_ResBlocks_spare_nothing': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["res"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '8_q_only_MBConvs_spare_nothing': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["mb"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '8_q_only_FusedMBConvs_spare_nothing': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["fmb"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '8_q_only_Attention_spare_nothing': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["att", "att@3", "att@5"],
        'spare_bottlenecks': False,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    
    ### Family ###
    '9_q_only_ResBlocks_spare_bottlenecks': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["res"],
        'spare_bottlenecks': True,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '9_q_only_MBConvs_spare_bottlenecks': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["mb"],
        'spare_bottlenecks': True,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '9_q_only_FusedMBConvs_spare_bottlenecks': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["fmb"],
        'spare_bottlenecks': True,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
    '9_q_only_Attention_spare_bottlenecks': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_names': ["att", "att@3", "att@5"],
        'spare_bottlenecks': True,
        'spare_attention_qkv': False,
        'spare_attention_scaling': False,
        'spare_attention_projection': False,
    },
}


REGISTERED_BACKBONE_DESCRIPTIONS = {
    "baseline": "baselines of the backbone, everything in FP32 or everything quantized",
    "1": "Early trial experiments",
    "2": "Early trial experiments",
    "3": "Aggersive quantization - Lift one STAGE back to FP32 at a time.",
    "4": "Aggresive quantization - Lift one BLOCK TYPE back to FP32 at a time.",
    "5": "Aggresive quantization - SPARE one special thing back to FP32 at a time.",
    "6": "Soft quantization - one STAGE at a time",
    "7": "Soft quantization - one STAGE at a time, plus save bottleneck for that stage",
    "8": "Soft quantization - one BLOCK TYPE at a time",
    "9": "Soft quantization - one BLOCK TYPE at a time, plus save bottleneck if in those block types",
}

SIMPLE_REGISTERED_BACKBONE_DESCRIPTIONS = {
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


SIMPLE_REGISTERED_BACKBONE_DESCRIPTIONS_DETAILED = {
    ### Just t''o provide types
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



'''
'3:0:2': /image_encoder/backbone/stages.3/op_list.0/main/point_conv/conv/Conv
'3:1:2': /image_encoder/backbone/stages.3/op_list.1/main/point_conv/conv/Conv
'3:2:2': /image_encoder/backbone/stages.3/op_list.2/main/point_conv/conv/Conv
'3:3:2': /image_encoder/backbone/stages.3/op_list.3/main/point_conv/conv/Conv
'3:4:2': /image_encoder/backbone/stages.3/op_list.4/main/point_conv/conv/Conv

'4:0:2': /image_encoder/backbone/stages.4/op_list.0/main/point_conv/conv/Conv
'4:1:6': /image_encoder/backbone/stages.4/op_list.1/local_module/main/point_conv/conv/Conv
'4:2:6': /image_encoder/backbone/stages.4/op_list.2/local_module/main/point_conv/conv/Conv
'4:3:6': /image_encoder/backbone/stages.4/op_list.3/local_module/main/point_conv/conv/Conv
'4:4:6': /image_encoder/backbone/stages.4/op_list.4/local_module/main/point_conv/conv/Conv

NAME_DICTIONARY_L0= {
    ### Name: Description, name in TRT, kernel shape, stride, dilation, input channels, putput channels
    '0:0:0': {
        'Input Conv (bottleneck)', 
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '0:1:0': {
        'ResBlock conv',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '0:1:1': {
        'Resblock conv no act',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '1:0:0': {
        'FMBConv 3x3 (bottleneck)',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '1:0:1': {
        'FMBConv 1x1 no act (bottleneck)',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '1:1:0': {
        'FMBConv 3x3',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '1:1:1': {
        'FMBConv 1x1 no act',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '2:0:0': {
        'FMBConv 3x3 (bottleneck)',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '2:0:1': {
        'FMBConv 1x1 no act (bottleneck)',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '2:1:0': {
        'FMBConv 3x3',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '2:1:1': {
        'FMBConv 1x1 no act',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '3:0:0': {
        'MBConv 1x1 no norm (bottleneck)', 
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '3:0:1': {
        'MBConv DWconv no norm (bottleneck)',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '3:0:2': {
        'MBConv 1x1 no act (bottleneck)',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '3:1:0': {
        'MBConv 1x1 no norm',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '3:1:1': {
        'MBConv DWconv no norm',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '3:1:2': {
        'MBConv 1x1 no act',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '3:2:0': {
        'MBConv 1x1 no norm',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '3:2:1': {
        'MBConv DWconv no norm',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '3:2:2': {
        'MBConv 1x1 no act',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '3:3:0': {
        'MBConv 1x1 no norm',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '3:3:1': {
        'MBConv DWconv no norm',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '3:3:2': {
        'MBConv 1x1 no act',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '3:4:0': {
        'MBConv 1x1 no norm',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '3:4:1': {
        'MBConv DWconv no norm',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '3:4:2': {
        'MBConv 1x1 no act',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:0:0': {
        'MBConv 1x1 no norm (bottleneck)',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:0:1': {
        'MBConv DWconv no norm (bottleneck)',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:0:2': {
        'MBConv 1x1 no act (bottleneck)',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:1:0': {
        'QKV-Conv 1x1',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:1:1': {
        'Scaling Conv 5x5, no norm, no act',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:1:2': {
        'Scaling Conv 1x1, no norm, no act',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:1:3': {
        'Projection conv 1x1',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:1:4': {
        'MBConv 1x1 no norm',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:1:5': {
        'MBConv DWconv no norm',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:1:6': {
        'MBConv 1x1 no act',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:2:0': {
        'QKV-Conv 1x1',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:2:1': {
        'Scaling Conv 5x5, no norm, no act',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:2:2': {
        'Scaling Conv 1x1, no norm, no act',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:2:3': {
        'Projection conv 1x1',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:2:4': {
        'MBConv 1x1 no norm',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:2:5': {
        'MBConv DWconv no norm',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:2:6': {
        'MBConv 1x1 no act',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:3:0': {
        'QKV-Conv 1x1',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:3:1': {
        'Scaling Conv 5x5, no norm, no act',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:3:2': {
        'Scaling Conv 1x1, no norm, no act',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:3:3': {
        'Projection conv 1x1',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:3:4': {
        'MBConv 1x1 no norm',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:3:5': {
        'MBConv DWconv no norm',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:3:6': {
        'MBConv 1x1 no act',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:4:0': {
        'QKV-Conv 1x1',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:4:1': {
        'Scaling Conv 5x5, no norm, no act',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:4:2': {
        'Scaling Conv 1x1, no norm, no act',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:4:3': {
        'Projection conv 1x1',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:4:4': {
        'MBConv 1x1 no norm',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:4:5': {
        'MBConv DWconv no norm',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
    '4:4:6': {
        'MBConv 1x1 no act',
        name,
        input channels,
        output channels,
        kernel,
        stride,
        dilatopn
        },
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
'''