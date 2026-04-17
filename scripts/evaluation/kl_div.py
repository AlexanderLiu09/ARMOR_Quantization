import os
import sys
print("pid", os.getpid())

import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb

from transformers import AutoModelForCausalLM

if __name__ == "__main__":
    print(os.getcwd())
    sys.path.append(os.getcwd())

from src.utils.model_utils import get_compressed_model_class
import src.utils.data_old as data_old


def _load_base(model_name, base_model_path):
    path = base_model_path if base_model_path is not None else model_name
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map="auto"
    )


def _load_quant(model_name, quant_model_path, load_custom_model):
    if load_custom_model:
        assert quant_model_path is not None, "quant_model_path required with --load_custom_model"
        return get_compressed_model_class(model_name).from_pretrained(
            quant_model_path, torch_dtype=torch.float16, device_map="auto"
        )
    assert quant_model_path is not None, "quant_model_path required"
    return AutoModelForCausalLM.from_pretrained(
        quant_model_path, torch_dtype=torch.float16, device_map="auto"
    )


@torch.no_grad()
def kl_eval_single_dataset(
    base_model,
    quant_model,
    model_name: str,
    dataset_name: str,
    seqlen: int = -1,
    log_wandb: bool = False,
):
    """Mean per-token KL(base || quant). Base is the reference distribution."""
    print(f"Evaluating KL on {dataset_name} ...")

    if seqlen == -1:
        seqlen = base_model.config.max_position_embeddings
        print("using model's max sequence length:", seqlen)

    testenc = data_old.get_loaders(
        dataset_name, nsamples=0, seqlen=seqlen, model=model_name, train_test="test"
    ).input_ids
    nsamples = testenc.numel() // seqlen

    kl_sum = 0.0
    top1_agree = 0
    total_tokens = 0
    per_seq_kl = []

    for i in tqdm.tqdm(range(nsamples), desc=f"KL {dataset_name}"):
        batch = testenc[:, i * seqlen : (i + 1) * seqlen].cuda()
        base_logits = base_model(batch)["logits"].float()
        quant_logits = quant_model(batch)["logits"].float()

        log_base = F.log_softmax(base_logits, dim=-1)
        log_quant = F.log_softmax(quant_logits, dim=-1)
        base_probs = log_base.exp()

        kl_tok = (base_probs * (log_base - log_quant)).sum(dim=-1)  # [B, T]
        kl_sum += kl_tok.sum().item()
        top1_agree += (base_logits.argmax(-1) == quant_logits.argmax(-1)).sum().item()
        total_tokens += kl_tok.numel()
        per_seq_kl.append(kl_tok.mean().item())

    mean_kl = kl_sum / total_tokens
    top1 = top1_agree / total_tokens
    q = np.quantile(per_seq_kl, [0.5, 0.9, 0.99, 1.0]).tolist()
    out = {
        "mean_kl": mean_kl,
        "median_seq_kl": q[0],
        "p90_seq_kl": q[1],
        "p99_seq_kl": q[2],
        "max_seq_kl": q[3],
        "top1_agreement": top1,
    }
    print(f"{dataset_name} KL: {out}")
    if log_wandb:
        wandb.log({f"kl/{dataset_name}/{k}": v for k, v in out.items()})
    return out


def main():
    parser = argparse.ArgumentParser(description="KL divergence between base and quantized model")
    parser.add_argument("--model_name", type=str, required=True, help="HF model name (tokenizer + base weights)")
    parser.add_argument("--base_model_path", type=str, default=None, help="Override path for base model")
    parser.add_argument("--quant_model_path", type=str, required=True, help="Path to quantized/compressed checkpoint")
    parser.add_argument("--load_custom_model", action="store_true", help="Load quant via get_compressed_model_class")
    parser.add_argument("--dataset_names", type=str, nargs="+",
                        choices=["wikitext2", "c4"], default=["wikitext2", "c4"])
    parser.add_argument("--seqlen", type=int, default=-1)
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--results_path", type=str, default="None")
    args = parser.parse_args()
    print("Arguments:", args)

    if args.log_wandb:
        info_path = os.path.join(os.path.dirname(args.quant_model_path), "wandb_info.yaml")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                w = yaml.safe_load(f)["wandb"]
            wandb.init(id=w["run_id"], project=w["project"], entity=w["entity"],
                       group=w.get("group"), name=w["name"], resume="must")
        else:
            print("warning, no wandb info found, initializing from scratch")
            wandb.init()

    base_model = _load_base(args.model_name, args.base_model_path)
    quant_model = _load_quant(args.model_name, args.quant_model_path, args.load_custom_model)
    base_model.eval(); quant_model.eval()

    results_path = None
    for dataset in args.dataset_names:
        out = kl_eval_single_dataset(
            base_model=base_model,
            quant_model=quant_model,
            model_name=args.model_name,
            dataset_name=dataset,
            seqlen=args.seqlen,
            log_wandb=args.log_wandb,
        )
        if args.save:
            results_path = (
                args.results_path if args.results_path != "None"
                else os.path.join(os.path.dirname(args.quant_model_path), "kl_results.yaml")
            )
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            results = {}
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    results = yaml.safe_load(f) or {}
            results[dataset] = {**out, "model_name": args.model_name, "seqlen": args.seqlen}
            with open(results_path, "w") as f:
                yaml.safe_dump(results, f)

    print(f"Evaluation completed. Results saved to {results_path if args.save else 'not saved'}")


if __name__ == "__main__":
    main()
