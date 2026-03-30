#!/bin/bash
# Same as ARMOR_quantized.bash but automatically uploads the compressed model
# to Box via rclone and stops the vast instance when compression finishes.

enviroment="ARMOR_main"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $enviroment

#defualt args
declare gpus=4,5,6,7 #change this for your machine
declare run_name="_run_name"
declare model="google/gemma-2-9b"
declare dataset_config="[{dataset_config:SlimPajama-627B,n_samples:128,ctx_len:8192}]"
declare block_size=128 #default block size for BlockPrune
declare n_iters=5000 #default number of iterations for BlockPrune
declare additional_args=""
declare quant_n_bits=8
declare quant_group_size=128
declare quant_start=500
declare vast_api_key=""
declare rclone_remote="UCLA_box"

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

stop_instance() {
    echo "Stopping vast instance in 30 seconds (Ctrl+C to cancel)..."
    sleep 30
    if [ -n "$vast_api_key" ]; then
        vastai stop instance "$VAST_CONTAINERLABEL" --api-key "$vast_api_key"
    else
        vastai stop instance "$VAST_CONTAINERLABEL"
    fi
}

upload() {
    local model_path=$1
    local box_path="ARMOR/${model_path}"

    echo "Uploading ${model_path} to ${rclone_remote}:${box_path} ..."
    rclone copy "$model_path" "${rclone_remote}:${box_path}" --progress
    if [ $? -ne 0 ]; then
        echo "ERROR: rclone upload failed."
        return 1
    fi
    echo "Upload complete."
}

trap stop_instance EXIT

original_run_name=$run_name
run_name="ARMOR_Q${quant_n_bits}/${block_size}_${n_iters}/${run_name}"


cmd="CUDA_VISIBLE_DEVICES=${gpus} python -u ParallelCompress.py \
    base_model=$model \
    log_wandb=True \
    compress=ARMOR_quantized \
    run_name=$run_name \
    compress.compression_config.block_diagonal_config.block_size=$block_size \
    +compress.compression_config.naive_compression_config.compression_config.pattern=[2,4] \
    compress.compression_config.training_config.n_iters=$n_iters \
    compress.compression_config.training_config.quant_n_bits=$quant_n_bits \
    compress.compression_config.training_config.quant_group_size=$quant_group_size \
    compress.compression_config.training_config.quant_qat_start_iter=$quant_start
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
exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "Compression failed with exit code $exit_code. Skipping upload."
    exit $exit_code
fi

model_output="models/${model}/compressed/${run_name}/model"
echo "Compression finished successfully."
upload "$model_output" || exit 1
