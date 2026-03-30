import os
import sys
print("pid", os.getpid())

CUDA_LAUNCH_BLOCKING = 1
import time

import torch
import random
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Literal

# from vector_quantizer import *
import tqdm

# from quant import *
import random
import numpy as np
import wandb


import yaml
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    print(os.getcwd())
    sys.path.append(os.getcwd())
    
from src.utils.model_utils import get_compressed_model_class
import src.utils.data_old as data_old
import src.utils.utils as utils


@torch.no_grad()
def ppl_eval_single_dataset(
    model,
    model_name: str,
    dataset_name: str,
    seqlen: int = -1,
    log_wandb: bool = False
) -> float:
    print("Evaluating ...")
    
    #if seqlen is not specified, use the model's max sequence length
    if seqlen == -1:
        seqlen = model.config.max_position_embeddings
        print("using model's max sequence length:", seqlen)
    testenc = data_old.get_loaders(
                    dataset_name,
                    nsamples=0,
                    seqlen=seqlen,
                    model=model_name,
                    train_test="test",
                )
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen
    nlls = []
    loss_fct = nn.CrossEntropyLoss()
    for i in tqdm.tqdm(range(nsamples), desc=f"Evaluating {dataset_name}"):
        batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].cuda()
        outs = model(batch)
        lm_logits = outs["logits"]
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
            :, 1:
        ].cuda()
        # print(shift_labels)
        # raise Exception("stop")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        # print("loss", loss)
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print(f"{dataset_name} Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f"perplexity/{dataset_name}": ppl.item()})

    return ppl.item()

def main():
    parser = argparse.ArgumentParser(description="Evaluate PPL on a dataset")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--model_path", type=str, required=False, help="Path to the model if different from model_name",
                        default=None)
    parser.add_argument("--load_custom_model", action='store_true', help="Load a custom model from the specified path")
    parser.add_argument("--dataset_names", type=str, nargs='+', help="List of dataset names to evaluate",
                        choices=["wikitext2","c4"], default=["wikitext2","c4"])
    parser.add_argument("--seqlen", type=int, default=-1, help="Sequence length (default: -1 for model's max)")
    parser.add_argument("--log_wandb", action='store_true', help="Log results to Weights & Biases")
    parser.add_argument("--save", action='store_true', help="Save results to a file")
    parser.add_argument("--results_path", type=str, default="None", help="Path to save results (if --save is True)")
    parser.add_argument("--verify_quant", action='store_true', help="Print dtype of sparse core weight matrices to verify quantization")
    args = parser.parse_args()
    
    print("Arguments:", args)
    if args.log_wandb:
        #if a model path is provided
        if args.model_path is not None and os.path.exists(os.path.join(os.path.dirname(args.model_path), "wandb_info.yaml")):
            
            with open(os.path.join(os.path.dirname(args.model_path), "wandb_info.yaml"), "r") as f:
                wandb_info = yaml.safe_load(f)["wandb"]
                run_id = wandb_info["run_id"]
                project = wandb_info["project"]
                entity = wandb_info["entity"]
                group = wandb_info.get("group", None)
                name = wandb_info["name"]
            # Initialize wandb
            wandb.init(
                id=run_id,
                project=project,
                entity=entity,
                group=group,
                name=name,
                resume="must",
            )
        else:
            print("warning, no wandb info found, initalizing from scratch")
            wandb.init()
    if args.load_custom_model:
        assert args.model_path is not None, "Model path must be specified when loading a custom model"
        model = get_compressed_model_class(args.model_name).from_pretrained(args.model_path,
                                                        torch_dtype=torch.float16,
                                                        device_map="auto")
        
    else:
        if args.model_path is not None:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
    print(f"Model loaded: {args.model_name} from {args.model_path if args.load_custom_model else 'default path'}")

    if args.verify_quant:
        for name, module in model.named_modules():
            if hasattr(module, 'naive_compression_module'):
                ncm = module.naive_compression_module
                if hasattr(ncm, 'X_int'):
                    print(f"  {name}: X_int.dtype={ncm.X_int.dtype}, quantized={ncm.quantized}")
                elif hasattr(ncm, 'X'):
                    print(f"  {name}: X.dtype={ncm.X.dtype}, quantized={ncm.quantized} (NOT quantized)")
            
    for dataset in args.dataset_names:
        print(f"Evaluating dataset: {dataset}")
        ppl = ppl_eval_single_dataset(
            model=model,
            model_name=args.model_name,
            dataset_name=dataset,
            seqlen=args.seqlen,
            log_wandb=args.log_wandb
        )
        
        if args.save:
            if args.results_path == "None":
                results_path = os.path.join(os.path.dirname(args.model_path) if args.model_path is not None else "results", 
                                            f"ppl_results.yaml")
            else:
                results_path = args.results_path
            
            #if the current results_path does not exist, create it
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            
            #if the file already exists, load it
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = yaml.safe_load(f)
            else:
                results = {}
            
            results[dataset] = {
                "ppl": ppl,
                "model_name": args.model_name,
                "seqlen": args.seqlen
            }
            with open(results_path, 'w') as f:
                yaml.safe_dump(results, f)
    print(f"Evaluation completed. Results saved to {results_path if args.save else 'not saved'}")
    
if __name__ == "__main__":
    main()