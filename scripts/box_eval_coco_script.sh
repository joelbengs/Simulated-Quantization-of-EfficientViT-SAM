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
models=("l0" "l1" "l2" "xl0" "xl1")
prompt_type=box

echo "Starting evaluation of models: ${models[*]} on prompt type: $prompt_type"

for model in "${models[@]}"
do
  # Run the evaluation command for the current model
  torchrun --nproc_per_node=2 \
  eval_sam_model.py \
  --dataset coco \
  --image_root coco/val2017 \
  --annotation_json_file coco/annotations/instances_val2017.json \
  --model $model \
  --prompt_type $prompt_type
done

echo "Finished evaluation of models: ${models[*]} on prompt type: $prompt_type"
# make it executable with the command chmod +x scriptname.sh
# run it with ./scriptname.sh
# 