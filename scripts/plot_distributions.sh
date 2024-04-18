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

echo "--------- STARTING SCRIPT ---------}"

echo "Plotting distributions of model L0"

torchrun --nproc_per_node=2 \
eval_sam_model_joel.py \
--dataset coco \
--image_root coco/val2017 \
--image_root_calibration coco/minitrain2017 \
--annotation_json_file coco/annotations/instances_val2017.json \
--model l0_quant \
--prompt_type box \
--backbone_version L0:-:-:- \
--limit_iterations 25 \
--quantize_W \
--plot_distributions \
--suppress_print \
--script_name $(basename $0 .sh) # removes the .sh extension and the directory scripts/
# --export_dataframe \
# --suppress_print \
# --plot_distributions \
# --quantize_method_W $qmw \
# --quantize_A \
# --print_torchinfo \
# --quantize_N \
# --quantize_method_A $qma \
# --quantize_method_N $qmn \
# --observer-method_A $oma \
# --observer-method_N $omn \

echo "Finished Distribution plotting of model L0"
# make it executable with the command chmod +x scriptname.sh
# run it with ./scriptname.sh