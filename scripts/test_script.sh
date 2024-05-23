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

 # choices=["minmax", "ema", "omse", "percentile"]


backbones=(
L0:all:all:all
)

model=(
"l0_quant"
)

observer_methods=(
ema
)


#WAAAAARNING THE FLAGS ARE SET TO TRUE 

echo "--------- STARTING SCRIPT L0 ---------}"
for obsmethod in "${observer_methods[@]}"
do
  for backbone_item in "${backbones[@]}"
  do
    echo " "
    echo "Model $model, backbone_version: $backbone_item" with observer method $obsmethod §
    # Run the evaluation command for the current model - with --quantize and configurations
    torchrun --nproc_per_node=2 \
    eval_sam_model_joel.py \
    --dataset coco \
    --image_root coco/val2017 \
    --dataset_calibration sa-1b \
    --image_root_calibration sa-1b \
    --annotation_json_file coco/annotations/instances_val2017.json \
    --model $model \
    --limit_iterations 4500 \
    --prompt_type box \
    --backbone_version $backbone_item \
    --quantize_W \
    --quantize_A \
    --observer_method_W $obsmethod \
    --observer_method_A $obsmethod \
    --calibration_mode_W channel_wise \
    --script_name $model
    # --limit_iterations 10 \
    # --export_dataframe \
    # --print_progress \
    # --plot_distributions \
    # --quantize_method_W $qmw \
    # --quantize_N \ # NOT NORMALLY USED
    # --quantize_A \
    # --observer_method_W $obsmethod \
    # --observer_method_A $obsmethod \
    # --print_torchinfo \
  done
done

backbones=(
L2:all_but_neck:all:all
)

model=(
"l2_quant"
)

echo "--------- STARTING SCRIPT L2 ---------}"
for backbone_item in "${backbones[@]}"
do
  echo " "
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
  --limit_iterations 4500 \
  --prompt_type box \
  --backbone_version $backbone_item \
  --quantize_W \
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

backbones=(
XL1:all_but_neck:all:all
)

model=(
"xl1_quant"
)

echo "--------- STARTING SCRIPT XL1 ---------}"
for backbone_item in "${backbones[@]}"
do
  echo " "
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
  --limit_iterations 2500 \
  --prompt_type box \
  --backbone_version $backbone_item \
  --quantize_W \
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