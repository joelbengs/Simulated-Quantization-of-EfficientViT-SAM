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



# OUTLIERS - To be verified

#backbones_L0=(
#L0:stage2:0:0
#L0:stage3:2:2
#L0:stage4:1:4
#L0:stage4:3:5
#)
#
#backbones_L2=(
#L2:stage3:1:1
#L2:stage4:1:4
#L2:stage4:2:4
#L2:stage4:7:4
#L2:stage4:8:4
#L2:neck:10:0
#L2:neck:11:0
#L2:neck:12:0
#)
#
#
#backbones_XL1=(
#XL1:stage2:0:0
#XL1:stage2:1:0
#XL1:stage3:0:0
#XL1:stage3:1:0
#XL1:stage3:2:0
#XL1:stage3:3:0
#XL1:stage3:4:0
#XL1:stage4:1:4
#XL1:stage5:0:0
#XL1:neck:13:0
#)
#

models=(
"l0_quant"
"l1_quant"
"l2_quant"
"xl0_quant"
"xl1_quant"
)

#WAAAAARNING THE FLAGS ARE SET TO TRUE no they not

echo "--------- STARTING SCRIPT L0 ---------}"
model=l0_quant
for backbone_item in "${backbones_L0[@]}"
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
  --calibration_mode_W channel_wise \
  --export_dataframe \
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

echo "--------- STARTING SCRIPT L2 ---------}"
model=l2_quant
for backbone_item in "${backbones_L2[@]}"
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
  --calibration_mode_W channel_wise \
  --export_dataframe \
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

echo "--------- STARTING SCRIPT XL1 ---------}"
model=xl1_quant
for backbone_item in "${backbones_XL1[@]}"
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
  --calibration_mode_W channel_wise \
  --export_dataframe \
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

echo "--------- EVERYTHING COMPLETED ---------}"

#for bbv in "${backbone_versions[@]}"
#do
#  echo "Model $model, backbone_version: $bbv" §
#  # Run the evaluation command for the current model - with --quantize and configurations
#  torchrun --nproc_per_node=2 \
#  eval_sam_model_joel.py \
#  --dataset coco \
#  --image_root coco/val2017 \
#  --dataset_calibration sa-1b \
#  --image_root_calibration sa-1b \
#  --annotation_json_file coco/annotations/instances_val2017.json \
#  --model $model \
#  --limit_iterations 10 \
#  --prompt_type box \
#  --backbone_version $bbv \
#  --quantize_W \
#  --quantize_N \
#  --quantize_A \
#  --script_name $(basename $0 .sh) # removes the .sh extension and the directory scripts/
#  # --limit_iterations 10 \
#  # --export_dataframe \
#  # --print_progress \
#  # --plot_distributions \
#  # --quantize_method_W $qmw \
#  # --quantize_A \
#  # --print_torchinfo \
#  # --quantize_N \
#  # --quantize_method_A $qma \
#  # --quantize_method_N $qmn \
#  # --observer-method_A $oma \
#  # --observer-method_N $omn \
#done

# Execute view_results.py
#python view_pickle_file.py --pickle_file_path results --script_name $(basename $0 .sh) --view_all_columns
