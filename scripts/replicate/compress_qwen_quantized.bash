#!/bin/bash

log_dir="./logs/"
datetime=$(date +"%Y%m%d_%H%M%S")

model_name=$1
gpus=$2 #change this for your machine
dataset=${3:-fineweb-edu} #dataset config name, default fineweb-edu
if [ -z "$4" ]; then
    num_processes=-1 #for inference
    parallelize=false
else
    num_processes=$4
    parallelize=true
fi


block_size=128
n_iters=20000
max_length=4096
quant_n_bits=4
quant_n_grid=100
quant_start=500




echo "Compressing model: $model_name"
echo "Generating data"
generate_cmd="CUDA_VISIBLE_DEVICES=${gpus} scripts/calibration_data_generation/generate.bash ${model_name} ${dataset} 128 8192"
echo "$generate_cmd"
eval "$generate_cmd"
if [ $? -ne 0 ]; then
    echo "Data generation failed for model: $model_name"
    return 1
fi
echo "generated data"
log_dir_use="${log_dir}/${model_name}/"
mkdir -p "$log_dir_use"
echo "Running quantized compression for model: $model_name"


scripts/compress/ARMOR_quantized.bash run_name=${datetime} \
    model="${model_name}" \
    block_size=$block_size \
    n_iters=$n_iters \
    dataset_config="[{dataset_config:${dataset},n_samples:128,ctx_len:8192}]" \
    gpus="${gpus}" \
    quant_n_bits=$quant_n_bits \
    quant_n_grid=$quant_n_grid \
    quant_start=$quant_start > "${log_dir_use}/ARMOR_Q${quant_n_bits}_${block_size}_${n_iters}.log" 2>&1
if [ $? -ne 0 ]; then
    echo "Compression failed for model: $model_name"
    exit 1
fi
echo "Compression completed for model: $model_name"

results_path="models/${model_name}/compressed/ARMOR_Q${quant_n_bits}/${block_size}_${n_iters}/${datetime}"




scripts/evaluation/pretrain_evaluation.bash \
    model_path="${results_path}" \
    model_name="$model_name" \
    generate_non_compressed_model=true \
    results_path="${results_path}/eval" \
    parallelize=$parallelize \
    num_processes=$num_processes \
    max_length=$max_length \
    gpus="$gpus" > "${log_dir_use}/Eval_ARMOR_Q${quant_n_bits}_${block_size}_${n_iters}.log" 2>&1
