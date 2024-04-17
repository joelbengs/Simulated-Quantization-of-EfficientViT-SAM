#!/bin/bash
# author: Joel Bengs

# Script for evaluating models  in SAM
# Box-Prompted Zero-Shot Instance Segmentation
# Ground Truth Bounding Box

# Get the number of CPU cores
nb_cpu_threads=$(nproc)

# Set the number of processes per node (you choose)
nproc_per_node=2

# Calculate the optimal number of OpenMP threads, according to a formula from github: https://github.com/pytorch/pytorch/issues/22260
export OMP_NUM_THREADS=$((nb_cpu_threads / nproc_per_node))
# Note: Time to execute was the same if this was =1 or =32

prompt_type=box

# Define the model family and prompt type 
# WARNING!! Models without _quant in the name will not use the correct backbone!!
models=(
"l0_quant"
"l1_quant"
"l2_quant"
"xl0_quant"
"xl1_quant"
)
models=(
"l0_quant"
)

backbone_versions=(
"FP32_baseline"
"INT8_baseline"
"3_q_all_but_stage0"
"3_q_all_but_stage1"
"3_q_all_but_stage2"
"3_q_all_but_stage3"
"3_q_all_but_stage4"
"3_q_all_but_stage5"
"3_q_all_but_neck"
"4_q_all_but_ResBlocks"
"4_q_all_but_MBConvs"
"4_q_all_but_FusedMBConvs"
"4_q_all_but_Attention"
"5_q_all_but_bottlenecks"
"5_q_all_but_attention_qkv"
"5_q_all_but_attention_scaling"
"5_q_all_but_attention_projection"
"6_q_only_stage0_spare_nothing"
"6_q_only_stage1_spare_nothing"
"6_q_only_stage2_spare_nothing"
"6_q_only_stage3_spare_nothing"
"6_q_only_stage4_spare_nothing"
"6_q_only_stage5_spare_nothing"
"6_q_only_neck_spare_nothing"
"7_q_only_stage0_spare_bottlenecks"
"7_q_only_stage1_spare_bottlenecks"
"7_q_only_stage2_spare_bottlenecks"
"7_q_only_stage3_spare_bottlenecks"
"7_q_only_stage4_spare_bottlenecks"
"7_q_only_stage5_spare_bottlenecks"
"7_q_only_neck_spare_bottlenecks"
"8_q_only_ResBlocks_spare_nothing"
"8_q_only_MBConvs_spare_nothing"
"8_q_only_FusedMBConvs_spare_nothing"
"8_q_only_Attention_spare_nothing"
"9_q_only_ResBlocks_spare_bottlenecks"
"9_q_only_MBConvs_spare_bottlenecks"
"9_q_only_FusedMBConvs_spare_bottlenecks"
"9_q_only_Attention_spare_bottlenecks"
)

backbone_versions=(
"INT8_baseline"
)

backbone_versions=(
L0:-:-:-
L0:x:x:x
L0:stage0:0:x
L0:stage0:1:x
L0:stage1:0:x
L0:stage1:1:x
L0:stage2:0:x
L0:stage2:1:x
L0:stage3:0:x
L0:stage3:1:x
L0:stage3:2:x
L0:stage3:3:x
L0:stage3:4:x
L0:stage4:0:x
L0:stage4:1:x
L0:stage4:2:x
L0:stage4:3:x
L0:stage4:4:x
L0:neck:0:x
L0:neck:1:x
L0:neck:2:x
L0:neck:3:x
L0:neck:4:x
L0:neck:5:x
L0:neck:6:x
L0:neck:7:x
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

backbone_versions=(
L0:-:-:-
)

echo "--------- STARTING SCRIPT ---------}"
for bbv in "${backbone_versions[@]}"
do
  for model in "${models[@]}"
  do
    echo "Model $model, backbone_version: $bbv" §
    # Run the evaluation command for the current model - with --quantize and configurations
    torchrun --nproc_per_node=2 \
    eval_sam_model_joel.py \
    --dataset coco \
    --image_root coco/val2017 \
    --image_root_calibration coco/minitrain2017 \
    --annotation_json_file coco/annotations/instances_val2017.json \
    --model $model \
    --prompt_type $prompt_type \
    --backbone_version $bbv \
    --limit_iterations 100 \
    --quantize_W \
    --plot_distributions \
    --suppress_print \
    --script_name $(basename $0 .sh) # removes the .sh extension and the directory scripts/
    # --quantize_method_W $qmw \
    #--plot_distributions \
    # --export_dataframe \
    # --quantize_A \
    # --print_torchinfo \
    # --quantize_N \
    # --quantize_method_A $qma \
    # --quantize_method_N $qmn \
    # --observer-method_A $oma \
    # --observer-method_N $omn \
    # --suppress_print \
  done
done

# Execute view_results.py
# python view_pickle_file.py --pickle_file_path results --script_name $(basename $0 .sh) --view_all_columns

echo "Finished $prompt_type --quanitze evaluation of models: ${models[*]}"
# make it executable with the command chmod +x scriptname.sh
# run it with ./scriptname.sh