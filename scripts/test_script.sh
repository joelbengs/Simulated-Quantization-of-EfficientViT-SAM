#!/bin/bash
# author: Joel Bengs

# Script for evaluating weight + activation quantization
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



backbones=(
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


model=(
"xl1_quant"
)

echo "--------- STARTING SCRIPT L0 ---------}"
for backbone_item in "${backbones[@]}"
do
  echo "Model $model, backbone_version: $backbone_item" §
  # Run the evaluation command for the current model - with --quantize and configurations
  torchrun --nproc_per_node=2 \
  eval_sam_model_joel.py \
  --dataset coco \
  --image_root coco/val2017 \
  --dataset_calibration sa-1b \
  --image_root_calibration sa-1b \
  --annotation_json_file coco/annotations/instances_val2017.json \
  --model $model \
  --limit_iterations 10 \
  --prompt_type box \
  --backbone_version $backbone_item \
  --quantize_W \
  --quantize_N \
  --quantize_A \
  --calibration_mode_W channel_wise \
  --script_name $model
  # --limit_iterations 10 \
  # --export_dataframe \
  # --print_progress \
  # --plot_distributions \
  # --quantize_method_W $qmw \
  # --quantize_A \
  # --print_torchinfo \
  # --quantize_N \
  # --quantize_method_A $qma \
  # --quantize_method_N $qmn \
  # --observer-method_A $oma \
  # --observer-method_N $omn \
done
