#!/bin/bash
# author: Joel Bengs
# Script for evaluating strategies

# Get the number of CPU cores
nb_cpu_threads=$(nproc)
# Set the number of processes per node (you choose)
nproc_per_node=2
# Calculate the optimal number of OpenMP threads, according to a formula from github: https://github.com/pytorch/pytorch/issues/22260
export OMP_NUM_THREADS=$((nb_cpu_threads / nproc_per_node))
# Note: Time to execute was the same if this was =1 or =32

backbones=(
any:all:all:all
#any:all_but_neck:all:all
)
# the protection of MBConvs is hardcoded into the ops.py definitions
# WAAAAARNING THE FLAGS ARE SET TO False

prompts=(
#box
box_from_detector
)

models=(
l0_quant
l1_quant
l2_quant
xl0_quant
xl1_quant
)

echo "--------- Benchmarking script on COCO, INT8 everything ---------}"

for backbone_item in "${backbones[@]}"
do
    for prompt in "${prompts[@]}"
    do
        for model in "${models[@]}"
        do
            echo " "
            # Run the evaluation command for the current model - with --quantize and configurations
            torchrun --nproc_per_node=2 \
            eval_sam_model_joel.py \
            --dataset coco \
            --image_root coco/val2017 \
            --dataset_calibration sa-1b \
            --image_root_calibration sa-1b \
            --annotation_json_file coco/annotations/lvis_v1_val.json \
            --annotation_json_file coco/annotations/instances_val2017.json \
            --source_json_file coco/source_json_file/coco_vitdet.json \
            --limit_iterations 4500 \
            --model $model \
            --prompt_type $prompt \
            --backbone_version $backbone_item \
            --quantize_W \
            --quantize_A \
            --script_name benchmark
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
            echo "$prompt COCO: $model, $backbone_item" §
        done
    done
done

echo "--------- Benchmarking script on LVIS, INT8 everything ---------}"

for backbone_item in "${backbones[@]}"
do
    for prompt in "${prompts[@]}"
    do
        for model in "${models[@]}"
        do
            echo " "
            # Run the evaluation command for the current model - with --quantize and configurations
            torchrun --nproc_per_node=2 \
            eval_sam_model_joel.py \
            --dataset lvis \
            --image_root coco \
            --dataset_calibration sa-1b \
            --image_root_calibration sa-1b \
            --annotation_json_file coco/annotations/lvis_v1_val.json \
            --source_json_file coco/source_json_file/lvis_vitdet.json \
            --limit_iterations 20\
            --model $model \
            --prompt_type $prompt \
            --backbone_version $backbone_item \
            --quantize_W \
            --quantize_A \
            --script_name benchmark
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
            echo "$prompt LVIS: $model, $backbone_item" §
        done
    done
done

echo "--------- Benchmarking complete ---------}"