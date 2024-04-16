# Backbone architectures for quantization experimentation
#for key in REGISTERED_BACKBONE_VERSIONS.keys(): print(key)

# Creates backbones for layerwise investigation of L0
def create_simple_backbone_versions_L0():
    stages_L0 = ["stage0", "stage1", "stage2", "stage3", "stage4", "neck"]
    block_depth_per_stage_L0 = [1,1,1,4,4,7] # the neck actually has 8 blocks, but subtract 1 to compensate for the lack of extra bottleneck block
    layer_depth_per_block_L0 = [
        [1,2], # stage0 has 1 input conv and 1 resblock, which has 2 Convs
        [2,2], # stage1 has 1 + 1 FusedMBConv, they have 2 convs each
        [2,2], # stage2 has 1 + 1 FusedMBConv, they have 2 convs each
        [3,3,3,3,3], # stage3 has 1 + 4 MBConv, they have 3 convs each
        [3,6,6,6,6], # stage4 has 1 MBConv (which has 3 convs)+ 4 efficentViT Modules, which have 6 conv layers each (counting the trailing MBconv)
        [1,1,1,2,2,2,2,1]  # the neck has 3 blocks of conv+upsample, one learnable layer each. It has 4 FusedMBconvs, and 1 output conv
    ]

    # Initialize a backbone which quantize all layers
    backbone_dict = {
        'L0:x:x:x': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_positions': [0,1,2,3,4,5,6,7,8,9],
        'layer_positions': [0,1,2,3,4,5,6,7,8,9],
        }
     }

    # block granularity:
    for s, block_depth in zip(stages_L0, block_depth_per_stage_L0):
        for bd in range(block_depth+1): # +1 to include the bottleneck block of each stage (the nech has none, which is compensated for above)
            key = f"L0:{s}:{bd}:x"
            value = {
                'stages': [s],
                'block_positions': [bd]
            }
            backbone_dict[key] = value

    # layer granularity
    for s, block_depth, layer_depth in zip(stages_L0, block_depth_per_stage_L0, layer_depth_per_block_L0):
        for bd in range(block_depth+1): # +1 to include the bottleneck of each stage
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

### Format: Model:Stage:Block:Layer
'''SIMPLE_REGISTERED_BACKBONE_VERSIONS = {
    'L0:x:x:x': {
        'stages': ["unknown", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "neck"],
        'block_positions': [0,1,2,3,4,5,6,7,8,9],
    },
    'L0:stage0:0:x': {'block_positions': [0], 'stages': ['stage0']},
    'L0:stage0:1:x': {'block_positions': [1], 'stages': ['stage0']},
    'L0:stage1:0:x': {'block_positions': [0], 'stages': ['stage1']},
    'L0:stage1:1:x': {'block_positions': [1], 'stages': ['stage1']},
    'L0:stage2:0:x': {'block_positions': [0], 'stages': ['stage2']},
    'L0:stage2:1:x': {'block_positions': [1], 'stages': ['stage2']},
    'L0:stage3:0:x': {'block_positions': [0], 'stages': ['stage3']},
    'L0:stage3:2:x': {'block_positions': [2], 'stages': ['stage3']},
    'L0:stage3:3:x': {'block_positions': [3], 'stages': ['stage3']},
    'L0:stage3:4:x': {'block_positions': [4], 'stages': ['stage3']},
    'L0:stage4:0:x': {'block_positions': [0], 'stages': ['stage4']},
    'L0:stage4:1:x': {'block_positions': [1], 'stages': ['stage4']},
    'L0:stage4:2:x': {'block_positions': [2], 'stages': ['stage4']},
    'L0:stage4:3:x': {'block_positions': [3], 'stages': ['stage4']},
    'L0:stage4:4:x': {'block_positions': [4], 'stages': ['stage4']},
    'L0:neck:0:x': {'block_positions': [0], 'stages': ['neck']},
    'L0:neck:1:x': {'block_positions': [1], 'stages': ['neck']},
    'L0:neck:2:x': {'block_positions': [2], 'stages': ['neck']},
    'L0:neck:3:x': {'block_positions': [3], 'stages': ['neck']},
    'L0:neck:4:x': {'block_positions': [4], 'stages': ['neck']},
    'L0:neck:5:x': {'block_positions': [5], 'stages': ['neck']},
    'L0:neck:6:x': {'block_positions': [6], 'stages': ['neck']},
    'L0:neck:7:x': {'block_positions': [7], 'stages': ['neck']},
    'L0:neck:8:x': {'block_positions': [8], 'stages': ['neck']},
}
'''