# Author: Joel Bengs. Apache-2.0 license
# Script for evaluating quantization strategies on EfficientViT-SAM

# Get the number of CPU cores
nb_cpu_threads=$(nproc)
# Set the number of processes per node (you choose)
nproc_per_node=2
# Calculate the optimal number of OpenMP threads, according to a formula from github: https://github.com/pytorch/pytorch/issues/22260
export OMP_NUM_THREADS=$((nb_cpu_threads / nproc_per_node))
# Note: Time to execute was the same if this was =1 or =32

backbones=(
any:all_but_neck:all:all
)

prompts=(
box
box_from_detector
)

models=(
l0_quant
l1_quant
l2_quant
xl0_quant
xl1_quant
)

echo "--------- Benchmarking Mix-MBC-Neck on COCO (remeber to toggle three flags in OPS.py) ---------"

for backbone_item in "${backbones[@]}"
do
    for prompt in "${prompts[@]}"
    do
        for model in "${models[@]}"
        do
            echo " "
            # Run the evaluation command for the current model - with --quantize and configurations
            torchrun --nproc_per_node=2 \
            eval_sam_quant_model.py \
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
            --script_name benchmark_coco_mbc-neck
            echo "$prompt COCO: $model, $backbone_item" §
        done
    done
done

echo "--------- Benchmarking Mix-MBC-Neck on LVIS (remeber to toggle three flags in OPS.py) ---------"

for backbone_item in "${backbones[@]}"
do
    for prompt in "${prompts[@]}"
    do
        for model in "${models[@]}"
        do
            echo " "
            # Run the evaluation command for the current model - with --quantize and configurations
            torchrun --nproc_per_node=2 \
            eval_sam_quant_model.py \
            --dataset lvis \
            --image_root coco \
            --dataset_calibration sa-1b \
            --image_root_calibration sa-1b \
            --annotation_json_file coco/annotations/lvis_v1_val.json \
            --source_json_file coco/source_json_file/lvis_vitdet.json \
            --limit_iterations 4500\
            --model $model \
            --prompt_type $prompt \
            --backbone_version $backbone_item \
            --quantize_W \
            --quantize_A \
            --script_name benchmark_lvis_mbc-neck
            echo "$prompt LVIS: $model, $backbone_item" §
        done
    done
done

echo "--------- Benchmarking complete ---------}"