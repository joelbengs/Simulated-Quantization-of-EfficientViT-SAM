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
#models=("l0_quant" "l1_quant" "l2_quant" "xl0_quant" "xl1_quant")
models=("l0_quant" "l2_quant" "xl1_quant")
prompt_type=box
#observer_method_W=("minmax" "ema" "omse" "percentile")
observer_method_W=("percentile")
backbone_version=('1a' '1b' '1c' '1d' '1e' '2a' '2b' '2c' '2d' '2e')

echo "--------- STARTING SCRIPT ---------}"
for model in "${models[@]}"
do
  for omw in "${observer_method_W[@]}"
  do
    echo "Model $model, observer: $omw, backbone_version: $backbone_version"
    # Run the evaluation command for the current model - with --quantize and configurations
    torchrun --nproc_per_node=2 \
    eval_sam_model_joel.py \
    --dataset coco \
    --image_root coco/val2017 \
    --image_root_calibration coco/minitrain2017 \
    --annotation_json_file coco/annotations/instances_val2017.json \
    --model $model \
    --prompt_type $prompt_type \
    --quantize_W \
    --observer_method_W $omw \
    --backbone_version $backbone_version \
    --limit_iterations 2500 \
    --export_dataframe \
    --suppress_print \
    --script_name $(basename $0 .sh) # removes the .sh extension and the directory scripts/
    # --quantize_method_W $qmw \
    # --quantize_A \
    # --quantize_N \
    # --quantize_method_A $qma \
    # --quantize_method_N $qmn \
    # --observer-method_A $oma \
    # --observer-method_N $omn \
      
  done
done

# Execute view_results.py
python view_pickle_file.py --pickle_file_path results --script_name $(basename $0 .sh) --view_all_columns

echo "Finished $prompt_type --quanitze evaluation of models: ${models[*]}"
# make it executable with the command chmod +x scriptname.sh
# run it with ./scriptname.sh