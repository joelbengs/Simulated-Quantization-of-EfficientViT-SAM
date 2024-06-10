# Author: Joel Bengs. Apache-2.0 license
# Script for extracting distributions using the observers
# The key difference is that the flag --plot_distributions is used

# Get the number of CPU cores
nb_cpu_threads=$(nproc)
# Set the number of processes per node (you choose)
nproc_per_node=2
# Calculate the optimal number of OpenMP threads, according to a formula from github: https://github.com/pytorch/pytorch/issues/22260
export OMP_NUM_THREADS=$((nb_cpu_threads / nproc_per_node))
# Note: Time to execute was the same if this was =1 or =32

models=(
l0_quant
l1_quant
l2_quant
xl0_quant
xl1_quant
)

echo "--------- STARTING SCRIPT ---------}"

for model in "${models[@]}"
do
    echo "Plotting distributions of models"
    torchrun --nproc_per_node=2 \
    eval_sam_quant_model.py \
    --dataset coco \
    --image_root coco/val2017 \
    --image_root_calibration coco/minitrain2017 \
    --annotation_json_file coco/annotations/instances_val2017.json \
    --model $model \
    --prompt_type box \
    --backbone_version any:none:none:none \
    --limit_iterations 25 \
    --quantize_W \
    --quantize_A \
    --plot_distributions \
    --script_name $(basename $0 .sh) # removes the .sh extension and the directory scripts/
done

echo "Finished Distribution plotting of all models"
# make it executable with the command chmod +x scriptname.sh
# run it with ./scriptname.sh