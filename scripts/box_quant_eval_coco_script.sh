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

# Define the model family and prompt type 
# WARNING!! Models without _quant in the name will not use the correct backbone!!
models=("l0_quant" "l1_quant" "l2_quant" "xl0_quant" "xl1_quant")
# models=("l0_quant" "xl1_quant")
prompt_type=box
#observer_method_W=("minmax" "ema" "omse" "percentile")
observer_method_W=("percentile")
# backbone_version=("1a" "1b" "1c" "1d" "1e" "2a" "2b" "2c" "2d" "2e")
# backbone_version=("1a")

# 25 versions
# backbone_version=("3_q_all" "3_q_all_but_stage0" "3_q_all_but_stage1" "3_q_all_but_stage2" "3_q_all_but_stage3" "3_q_all_but_stage4" "3_q_all_but_stage5" "3_q_all_but_bottlenecks" "3_q_all_but_qkv" "4_q_all_but_stage3_bottlenecks" "4_q_all_but_stage3_qkv" "4_q_all_but_stage3_scaling" "4_q_all_but_stage3_projection" "4_q_all_but_stage3_bottleneck_qkv" "4_q_all_but_stage3_bottleneck_scaling" "4_q_all_but_stage3_bottleneck_projection" "4_q_all_but_stage3_bottleneck_qkv_scaling_projection" "5_q_all_but_ResBlocks" "5_q_all_but_MBConvs" "5_q_all_but_FusedMBConvs" "5_q_all_but_Attention" "5_q_only_ResBlocks" "5_q_only_MBConvs" "5_q_only_FusedMBConvs" "5_q_only_Attention")
# backbone_version=("3_q_all_but_stage1" "3_q_all_but_stage2" "3_q_all_but_stage3" "3_q_all_but_stage4" "3_q_all_but_stage5" "3_q_all_but_bottlenecks" "3_q_all_but_qkv" "4_q_all_but_stage3_bottlenecks" "4_q_all_but_stage3_qkv" "4_q_all_but_stage3_scaling" "4_q_all_but_stage3_projection" "4_q_all_but_stage3_bottleneck_qkv" "4_q_all_but_stage3_bottleneck_scaling" "4_q_all_but_stage3_bottleneck_projection" "4_q_all_but_stage3_bottleneck_qkv_scaling_projection" "5_q_all_but_ResBlocks" "5_q_all_but_MBConvs" "5_q_all_but_FusedMBConvs" "5_q_all_but_Attention" "5_q_only_ResBlocks" "5_q_only_MBConvs" "5_q_only_FusedMBConvs" "5_q_only_Attention")
backbone_version=("FP32_baseline" "INT8_baseline" "5_q_only_MBConvs")

echo "--------- STARTING SCRIPT ---------}"
for bbv in "${backbone_version[@]}"
do
  for model in "${models[@]}"
  do
    for omw in "${observer_method_W[@]}"
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
      --observer_method_W $omw \
      --backbone_version $bbv \
      --limit_iterations 3 \
      --quantize_W \
      --export_dataframe \
      --suppress_print \
      --script_name $(basename $0 .sh) # removes the .sh extension and the directory scripts/
      # --quantize_method_W $qmw \
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
done

# Execute view_results.py
python view_pickle_file.py --pickle_file_path results --script_name $(basename $0 .sh) --view_all_columns

echo "Finished $prompt_type --quanitze evaluation of models: ${models[*]}"
# make it executable with the command chmod +x scriptname.sh
# run it with ./scriptname.sh