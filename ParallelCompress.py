import torch.multiprocessing as mp
import os
import glob
import argparse
import yaml
import queue
import torch
import torch.nn as nn
import tqdm
import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

import accelerate
import traceback
from transformers import AutoConfig, AutoModelForCausalLM
from typing import Dict, Any, List, Tuple, Optional, Callable

from src.utils.utils import *
from src.utils.model_utils import create_compressed_model
from src.data import *
from accelerate import infer_auto_device_map, dispatch_model
import numpy as np

#set to be deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def compression_worker(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    gpu_queue: mp.Queue,
    wandb_queue: mp.Queue,
    lock: "mp.synchronize.Lock",
    stop_event: "mp.synchronize.Event",
):
    """Quantizes a layer of the model

    Args:
        task_queue (mp.Queue): task queue, each element consits of a tuple of (layer_name[str],
        config[DictConfig], calibration_paths
        result_queue (mp.Queue): the results queue, each element is a tuple of
        (
                    layer_name[str],
                    save_path[str],
                    (n_bits[int], n_params[int]) if quantizing, (n_nonzero[int], n_params[int]) if pruning
        )
        gpu_queue (mp.Queue): a queue of the available GPU ids
        lock (mp.synchronize.Lock): a lock for accessing the GPU queue
        stop_event (mp.Event): event to stop the worker if a exception is raised
    """
    wandb_initialized = False
    while True:
        try:
            # Get next task (non-blocking)
            layer_name, cfg, calibration_paths, wandb_info = task_queue.get_nowait()
            use_wandb, wandb_run_info = wandb_info
            # if use_wandb and not wandb_initialized:
            #     # Initialize wandb only once
            #     with lock:
            #         print("Initializing wandb for run", wandb_run_info)
            #         wandb.init(
            #             project=wandb_run_info["project"],
            #             id=wandb_run_info["run_id"],
            #             resume="must",
            #         )
            #         wandb_initialized = True
            #         print("Wandb initialized for run", wandb_run_info["run_id"])
            # layer name is expected to be of the form layer_{i}/{self_attn|mlp}.{q|k|v|o|up|down|gate}_proj
        except queue.Empty:
            # No more tasks, exit
            return

        # Get an available GPU
        with lock:
            gpu_id = gpu_queue.get()

        try:
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)
            seed(cfg.seed, seed_all=True)

            # load the original weight
            original_weight_data = torch.load(
                os.path.join(cfg.weight_path, layer_name + ".pt"), map_location=device
            )
            weight = original_weight_data["weight"]
            bias = original_weight_data.get("bias", None)
            
            original_dtype = weight.dtype

            weight = weight.to(torch.float32)
            if bias is not None:
                bias = bias.to(torch.float32)

            # create the compression module
            compression_module = instantiate(
                cfg.compress.init_config,
                weight = weight,
                bias = bias,
            )
            if use_wandb:
                compression_module.enable_wandb(
                    metric_name=f"proxy_losses/{layer_name}",
                    step_metric="iteration",
                    wandb_queue=wandb_queue,
                )


            # load the calibration data 
            
            if cfg.calibration_data_type == "hessian_diag":
                total_samples = sum([
                    n_samples for _, n_samples in calibration_paths
                ])
                hessianDiag = torch.load(
                    os.path.join(calibration_paths[0][0], layer_name + ".pt"),
                    map_location=device,
                )["hessianDiag"] * calibration_paths[0][1] / total_samples
                for calibration_path, n_samples in calibration_paths[1:]:
                    hessianDiag += torch.load(
                        os.path.join(calibration_path, layer_name + ".pt"),
                        map_location=device
                    )["hessianDiag"] * n_samples / total_samples
                
                compression_module.hessianDiag = hessianDiag.to(torch.float32)
                compression_module.validate_hessianDiag()
                
            elif cfg.calibration_data_type == "hessian":
                total_samples = sum([
                    n_samples for _, n_samples in calibration_paths
                ])
                hessian = torch.load(
                    os.path.join(calibration_paths[0][0], layer_name + ".pt"),
                    map_location=device,
                )["hessian"] * calibration_paths[0][1] / total_samples
                for calibration_path, n_samples in calibration_paths[1:]:
                    hessian += torch.load(
                        os.path.join(calibration_path, layer_name + ".pt"),
                        map_location=device
                    )["hessian"] * n_samples / total_samples
                compression_module.hessian = hessian.to(torch.float32)
            else:
                raise ValueError("Unknown calibration data type {}".format(cfg.calibration_data_type))

            
            # compress the layer
            if cfg.resume_layerwise and os.path.exists(os.path.join(cfg.temp_path, layer_name + ".pt")):
                compression_module.blank_recreate(**cfg.compress.compression_config)
                compression_module.load_state_dict(torch.load(
                    os.path.join(cfg.temp_path, layer_name + ".pt"),
                    map_location=device
                ))
                compression_module.to(original_dtype)
            else:
                if cfg.iter_sweep:
                    sweep_save_path = os.path.join(
                        cfg.save_path, f"iter_sweep/{layer_name}/"
                    )
                    os.makedirs(sweep_save_path, exist_ok=True)
                    sweep_save_path = os.path.join(sweep_save_path, "iter_{iter}.pt")
                    
                    compression_module.compress(
                        training_config_overrides = {"save_path": sweep_save_path, "save_freq": cfg.iter_sweep_freq},
                        **cfg.compress.compression_config)
                else:
                    compression_module.compress(**cfg.compress.compression_config)
                
                # save the compression module to the temp path
                compression_module.to(original_dtype)

                if cfg.verbose:
                    print(
                        f"Compression module {layer_name} created, average unweighted l2 distortion: {compression_module.get_reconstruction_error()}, average weighted l2 distortion: {compression_module.get_reconstruction_error(error_weight = hessianDiag)}",
                        flush=True,
                    )
                # save the state dict
                state_dict = compression_module.state_dict()

                os.makedirs(
                    os.path.dirname(os.path.join(cfg.temp_path, layer_name + ".pt")),
                    exist_ok=True,
                )
                torch.save(state_dict, os.path.join(cfg.temp_path, layer_name + ".pt"))
                #delete everything and clean
                del state_dict
            # calculate the number of parameters
            compression_measure = ()
            if compression_module.compression_measure == "bits":
                compression_measure = (
                    compression_module.compression_measure,
                    compression_module.get_n_bits(),
                    compression_module.get_n_original_parameters(),
                )
            elif compression_module.compression_measure == "parameters":
                compression_measure = (
                    compression_module.compression_measure,
                    compression_module.get_n_nonzero(),
                    compression_module.get_n_original_parameters(),
                )
            else:
                raise ValueError(
                    f"Unknown compression measure {compression_module.compression_measure}"
                )
            
            del weight
            del compression_module
            if hasattr(cfg, "hessian_path"):
                del hessian
            if hasattr(cfg, "hessianDiag_path"):
                del hessianDiag
            clean()
            print(get_gpu_memory(device, return_str=True))
            # Put the GPU back in the queue
            with lock:
                gpu_queue.put(gpu_id)

            # Put the result in the result queue
            result_queue.put(
                (
                    layer_name,
                    os.path.join(cfg.temp_path, layer_name + ".pt"),
                    compression_measure,
                )
            )
            
        except Exception as e:
            print(
                f"========================= Error in quantization of {layer_name} ========================="
            )
            # print the error and the traceback
            print(e)
            traceback.print_exc()
            # Set the stop event to signal the main process
            stop_event.set()
            raise e
        # finally:
        #     # Put the GPU back in the queue
        #     with lock:
        #         gpu_queue.put(gpu_id)

        #     # Put the result in the result queue
        #     result_queue.put(
        #         (
        #             layer_name,
        #             os.path.join(cfg.temp_path, layer_name + ".pt"),
        #             compression_measure,
        #         )
        #     )


@hydra.main(version_base=None, config_path="./config", config_name="compress")
def main(cfg: DictConfig):
    #log to wandb if needed
    try:
        if cfg.log_wandb:
            run = wandb.init(
                project="ParallelCompress-1shot",
                config=OmegaConf.to_container(cfg, resolve=True),
                name=cfg.run_name,
                group=cfg.base_model.split("/")[-1],
            )
            run_info =  {"run_id": run.id, "project": run.project, "name": run.name, "entity": run.entity,
                         "group": run.group}
            #save the run information
            os.makedirs(cfg.save_path, exist_ok=True)
            with open(os.path.join(cfg.save_path, "wandb_info.yaml"), "w") as f:
                yaml.dump(
                    {
                        "wandb": run_info,
                    },
                    f,
                )
            use_wandb = True
        else:
            use_wandb = False
            run_info = None
            
            
        #print the config
        print(OmegaConf.to_yaml(cfg))
        #save the config
        os.makedirs(cfg.save_path, exist_ok=True)
        with open(os.path.join(cfg.save_path, "config.yaml"), "w") as f:
            yaml.dump(
                OmegaConf.to_container(cfg, resolve=True),
                f,
            )
        # raise ValueError("This script is not ready to run yet, please use the new compress.py script instead")
        # create our list of tasks
        weight_paths = glob.glob(os.path.join(cfg.weight_path, "*/*.pt"))
        print("n weights found", len(weight_paths))
        
        #parse the datasets
        calibration_paths = []
        for dataset in cfg.datasets:
            print("dataset", dataset)
            #load the dataset config
            if dataset["ctx_len"] < 0:
                print("for dataset", dataset["dataset_config"], "using model.config.max_position_embeddings=", cfg.model.config.max_position_embeddings)
                dataset["ctx_len"] = cfg.model.config.max_position_embeddings
                
            with open(os.path.join(cfg.dataset_config_base_path, dataset["dataset_config"] + ".yaml"), "r") as f:
                dataset_config_raw = yaml.safe_load(f)
                dataset_config = instantiate(dataset_config_raw,
                                            n_samples=dataset["n_samples"],
                                                ctx_len=dataset["ctx_len"])
                
            
            calibration_path = os.path.join(cfg.calibration_data_path,
                                            create_save_dir(dataset_config, seed=cfg.seed))
            
            #check that the config is the same as the one in the dataset config
            with open(os.path.join(calibration_path, "dataset_config.yaml"), "r") as f:
                assert dataset_config_raw == yaml.safe_load(f), "Dataset config does not match the one in the calibration path"
            #check that all the weights exists
            for weight_path in weight_paths:
                check_path = weight_path.replace(cfg.weight_path, calibration_path)
                assert os.path.exists(check_path), f"Calibration path {check_path} does not exist"
            calibration_paths.append([calibration_path, dataset["n_samples"]])

        
        
        os.makedirs(cfg.save_path, exist_ok=True)
        print(f"compressing model {cfg.base_model}")
        devices = torch.cuda.device_count()
        print("devices available:", devices)
        gpu_ids = list(range(devices))
        print(f"Available GPUs: {gpu_ids}")

        # Create queues for tasks and results
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        gpu_queue = mp.Queue()
        wandb_queue = mp.Queue()
        lock = mp.Lock()
        for gpu_id in gpu_ids:
            gpu_queue.put(gpu_id)


        # create the task queue
        for weight_path in weight_paths:
            layer_name = weight_path.replace(cfg.weight_path, "").replace(".pt", "")[1:]
            task_queue.put((layer_name, cfg, calibration_paths, (use_wandb, run_info)))
            if use_wandb:
                #define the metrics
                run.define_metric(
                    f"proxy_losses/{layer_name}",
                    step_metric="iteration"
                )

        # Create a stop event
        stop_event = mp.Event()

        # Create a pool of workers
        num_workers = min(len(gpu_ids), mp.cpu_count())
        processes = []
        print(f"Starting {num_workers} workers")
        for _ in range(num_workers):
            p = mp.Process(
                target=compression_worker,
                args=(task_queue, result_queue, gpu_queue, wandb_queue, lock, stop_event),
            )
            p.start()
            processes.append(p)

        # Create a progress bar
        pbar = tqdm.tqdm(total=len(weight_paths), desc="Compressing layers")

        checkpoints_dict: Dict[str, str] = {}
        running_first = 0
        running_params = 0

        # Process results
        tasks_done = 0
        while tasks_done < len(weight_paths):
            if use_wandb:
                try:
                    wandb_log = wandb_queue.get(timeout=1)
                    wandb.log(wandb_log)
                    #get everything elese from the queue until it is empty
                    while True:
                        try:
                            wandb_log = wandb_queue.get_nowait()
                            wandb.log(wandb_log)
                        except queue.Empty:
                            break
                except queue.Empty:
                    pass
                
            try:
                layer_name, save_path, (compression_measure, first, n_params) = result_queue.get(timeout=1)
                # print(f"Layer {layer_name} quantized and saved to {save_path}")
                checkpoints_dict[layer_name] = save_path
                running_first += first
                running_params += n_params
                tasks_done += 1
                pbar.update(1)
            except queue.Empty:
                pass

            if stop_event.is_set():
                print("Stopping all workers due to error")
                break
        pbar.close()

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Check if any process raised an exception
        if stop_event.is_set():
            print("One or more processes raised an exception. Exiting.")
            for p in processes:
                p.terminate()
            return

        # print out the results
        print("=" * 10, "Compression Level", "=" * 10)
        print(f"Total number of parameters: {human_format(running_params)}")
        if compression_measure == "bits":
            print(f"Total number of bytes: {human_format(running_first)}")
            print(f"Average bits per parameter: {round(running_first/running_params, 4)}")
            compression_log = {
                        "compression_measure": compression_measure,
                        "total_bytes": running_first,
                        "total_parameters": running_params,
                        "average_bits_per_parameter": round(running_first/running_params, 4),
                    }
        elif compression_measure == "parameters":
            print(f"Total number of non-zero parameters: {human_format(running_first)}")
            print(f"Actual Pruning fraction: {round(running_first/running_params, 4)}")
            compression_log = {
                        "compression_measure": compression_measure,
                        "total_nonzero_parameters": running_first,
                        "total_parameters": running_params,
                        "actual_pruning_fraction": round(running_first/running_params, 4),
                    }
            
        else:
            raise ValueError(f"Unknown compression measure {compression_measure}")  

        with open(os.path.join(cfg.save_path, "compression.yaml"), "w") as f:
                yaml.dump(
                    compression_log,
                    f,
                )
        if use_wandb:
            # log the compression level to wandb
            wandb.log(compression_log)
            # finish the run
            run.finish()
            
                
        print(f"=" * 25)

        # Reload and Save in hf format

        orig_config = AutoConfig.from_pretrained(
            cfg.base_model, dtype="auto", device_map="cpu", attn_implementation="sdpa"
        )
        orig_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            config=orig_config,
            torch_dtype="auto",
            device_map="cpu",
        )

        compressed_model = create_compressed_model(cfg.base_model, cfg, orig_config)
        compressed_model.load_state_dict(orig_model.state_dict(), strict=False)

        del orig_model

        for layer_name, save_path in tqdm.tqdm(
            checkpoints_dict.items(), desc="Loading compressed layers"
        ):
            # now split by /
            layer_name = layer_name.split("/")[-2:]
            # from the first part, we can get which layer it is
            i_layer = int(layer_name[0].replace("layer_", ""))
            # from the second part we can get which module (self_attn, mlp, etc) and which layer it is
            submodule_name, linear_name = layer_name[1].split(".")

            # now we get the right module
            layer = getattr(
                getattr(compressed_model.model.layers[i_layer], submodule_name), linear_name
            )
            # record the original dtype
            orig_dtype = next(layer.parameters()).dtype
            orig_device = next(layer.parameters()).device
            # load the state dict
            state_dict = torch.load(save_path, map_location=orig_device)
            # If the checkpoint was quantized, the naive_compression_module will have
            # X_int/scale/zero_point instead of X. Prepare the module to match.
            if "naive_compression_module.X_int" in state_dict:
                ncm = layer.naive_compression_module
                if hasattr(ncm, 'X'):
                    del ncm.X
                ncm.register_buffer("X_int", torch.empty(0))
                ncm.register_buffer("scale", torch.empty(0))
                ncm.register_buffer("zero_point", torch.empty(0))
                ncm.quantized = True
            layer.load_state_dict(state_dict)
            layer.to(orig_dtype)
            # delete the state dict to save memory
            del state_dict
        clean()
        # save the model
        compressed_model.save_pretrained(os.path.join(cfg.save_path, "model"))
        
        if cfg.resume_layerwise:
            # cleaned up all the temporary files now since its safe
            for save_path in checkpoints_dict.values():
                os.remove(save_path)
            print("done, cleaned up all the temporary files")
        
        
    finally:
        # pass
        # Clean up temporary files
        if not cfg.resume_layerwise:
            for save_path in checkpoints_dict.values():
                os.remove(save_path)
            print("done, cleaned up all the temporary files")
        else:
            print("not cleaning up temporary files since resume_layerwise is True")
            

if __name__ == "__main__":
    main()
