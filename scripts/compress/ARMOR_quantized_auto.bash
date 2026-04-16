#!/bin/bash

enviroment="ARMOR_main"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $enviroment

#defualt args
declare gpus=4,5,6,7 #change this for your machine
declare run_name="_run_name"
declare model="google/gemma-2-9b"
declare dataset_config="[{dataset_config:SlimPajama-627B,n_samples:128,ctx_len:8192}]"
declare block_size=128 #default block size for block-diagonal A/B
declare n_iters=5000 #default number of iterations
declare additional_args=""
declare quant_n_bits=4 #INT precision for the quantized sparse core
declare groupsize=128 #quantization groupsize (must divide d_in and be a multiple of sparse group)
declare sparse_core_step_select="gradient_all_random" #group selection strategy

#loop through the args
for arg in "$@"
do
    # Check if the argument is in key=value format
    if [[ "$arg" == *"="* ]]; then
        # Split into key and value
        key="${arg%%=*}"
        value="${arg#*=}"

        # 3. Use 'declare' again to update the variable
        # This will only affect variables already declared above.
        # For safety, you can check if the variable exists first.
        if declare -p "$key" &>/dev/null; then
            declare "$key=$value"
            echo "Updated argument '$key' to '$value'"
        else
            echo "ERROR: Unknown argument '$key'"
            exit 1
        fi
    fi
done

original_run_name=$run_name
run_name="ARMOR_Q${quant_n_bits}/${block_size}_${n_iters}/${run_name}"


cmd="CUDA_VISIBLE_DEVICES=${gpus} python -u ParallelCompress.py \
    base_model=$model \
    log_wandb=True \
    compress=ARMOR_quantized \
    run_name=$run_name \
    compress.compression_config.block_diagonal_config.block_size=$block_size \
    compress.compression_config.naive_compression_config.compression_config.quant_precision=$quant_n_bits \
    compress.compression_config.naive_compression_config.compression_config.groupsize=$groupsize \
    compress.compression_config.training_config.n_iters=$n_iters \
    compress.compression_config.training_config.sparse_core_step_select=$sparse_core_step_select \
     \"datasets=${dataset_config}\""

#split the additional args by space and add them to the command
for arg in $additional_args
do
    cmd+=" $arg"
done

echo "Command to run: $cmd"



source $(conda info --base)/etc/profile.d/conda.sh
conda activate $enviroment
eval $cmd
compress_exit_code=$?

#auto-stop the vast.ai instance once compression has finished
echo "Compression finished with exit code ${compress_exit_code}. Stopping vast.ai instance..."
if [ -n "$VAST_CONTAINERLABEL" ]; then
    instance_id="${VAST_CONTAINERLABEL#C.}"
    vastai stop instance "$instance_id"
else
    echo "WARNING: VAST_CONTAINERLABEL not set; cannot determine vast.ai instance ID. Skipping auto-stop."
fi

exit $compress_exit_code
