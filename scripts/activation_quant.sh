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
L0:stage4:1:6
L0:stage4:2:0
L0:stage4:2:1
L0:stage4:2:2
L0:stage4:2:3
L0:stage4:2:4
L0:stage4:2:5
L0:stage4:2:6
L0:stage4:3:0
L0:stage4:3:1
L0:stage4:3:2
L0:stage4:3:3
L0:stage4:3:4
L0:stage4:3:5
L0:stage4:3:6
L0:stage4:4:0
L0:stage4:4:1
L0:stage4:4:2
L0:stage4:4:3
L0:stage4:4:4
L0:stage4:4:5
L0:stage4:4:6
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
L0:stage4:1:3
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
    --dataset_calibration sa-1b \
    --image_root_calibration sa-1b \
    --annotation_json_file coco/annotations/instances_val2017.json \
    --model $model \
    --limit_iterations 10 \
    --prompt_type box \
    --backbone_version $bbv \
    --quantize_W \
    --quantize_N \
    --quantize_A \
    --script_name $(basename $0 .sh) # removes the .sh extension and the directory scripts/
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
done

# Execute view_results.py
#python view_pickle_file.py --pickle_file_path results --script_name $(basename $0 .sh) --view_all_columns