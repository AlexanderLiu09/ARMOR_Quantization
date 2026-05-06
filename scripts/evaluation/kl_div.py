import os
import sys
import gc
import tempfile
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


def _device_map(gpu):
    return "auto" if gpu is None else {"": int(gpu)}


def _load_base(model_name, base_model_path, gpu=None):
    path = base_model_path if base_model_path is not None else model_name
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map=_device_map(gpu)
    )


def _load_quant(model_name, quant_model_path, load_custom_model, gpu=None):
    assert quant_model_path is not None, "quant_model_path required"
    if load_custom_model:
        return get_compressed_model_class(model_name).from_pretrained(
            quant_model_path, torch_dtype=torch.float16, device_map=_device_map(gpu)
        )
    return AutoModelForCausalLM.from_pretrained(
        quant_model_path, torch_dtype=torch.float16, device_map=_device_map(gpu)
    )


def _model_device(model):
    return next(model.parameters()).device


def _get_testenc(model_name, dataset_name, seqlen):
    return data_old.get_loaders(
        dataset_name, nsamples=0, seqlen=seqlen, model=model_name, train_test="test"
    ).input_ids


def _reduce(kl_sum, top1_agree, total_tokens, per_seq_kl):
    q = np.quantile(per_seq_kl, [0.5, 0.9, 0.99, 1.0]).tolist()
    return {
        "mean_kl": kl_sum / total_tokens,
        "median_seq_kl": q[0],
        "p90_seq_kl": q[1],
        "p99_seq_kl": q[2],
        "max_seq_kl": q[3],
        "top1_agreement": top1_agree / total_tokens,
    }


@torch.no_grad()
def kl_eval_dual_gpu(base_model, quant_model, model_name, dataset_name, seqlen, log_wandb, chunk_size=512):
    """KL(base || quant) with each model pinned to its own device.

    Logits stay in fp16 until we pull them into a fp32 log-softmax chunk of
    ``chunk_size`` tokens, so peak memory is ~2 * chunk_size * vocab * 4 bytes
    per sequence rather than the full 2 * seqlen * vocab * 4.
    """
    print(f"[dual_gpu] Evaluating KL on {dataset_name} ...")
    if seqlen == -1:
        seqlen = base_model.config.max_position_embeddings
        print("using model's max sequence length:", seqlen)

    base_dev = _model_device(base_model)
    quant_dev = _model_device(quant_model)
    testenc = _get_testenc(model_name, dataset_name, seqlen)
    nsamples = testenc.numel() // seqlen

    kl_sum, top1_agree, total_tokens = 0.0, 0, 0
    per_seq_kl = []
    for i in tqdm.tqdm(range(nsamples), desc=f"KL {dataset_name}"):
        batch = testenc[:, i * seqlen : (i + 1) * seqlen]
        base_logits = base_model(batch.to(base_dev))["logits"]        # fp16, base_dev
        quant_logits = quant_model(batch.to(quant_dev))["logits"]     # fp16, quant_dev

        base_argmax = base_logits.argmax(-1)
        quant_argmax = quant_logits.argmax(-1).to(base_dev)
        top1_agree += (base_argmax == quant_argmax).sum().item()

        T = base_logits.shape[1]
        seq_kl_sum, seq_tokens = 0.0, 0
        for s in range(0, T, chunk_size):
            e = min(s + chunk_size, T)
            bl = base_logits[:, s:e, :].float()
            ql = quant_logits[:, s:e, :].to(base_dev).float()
            lb = F.log_softmax(bl, dim=-1)
            lq = F.log_softmax(ql, dim=-1)
            kl_chunk = (lb.exp() * (lb - lq)).sum(dim=-1)
            seq_kl_sum += kl_chunk.sum().item()
            seq_tokens += kl_chunk.numel()
            del bl, ql, lb, lq, kl_chunk

        kl_sum += seq_kl_sum
        total_tokens += seq_tokens
        per_seq_kl.append(seq_kl_sum / seq_tokens)

        del base_logits, quant_logits, base_argmax, quant_argmax
        torch.cuda.empty_cache()

    out = _reduce(kl_sum, top1_agree, total_tokens, per_seq_kl)
    print(f"{dataset_name} KL: {out}")
    if log_wandb:
        wandb.log({f"kl/{dataset_name}/{k}": v for k, v in out.items()})
    return out


@torch.no_grad()
def cache_base_topk(base_model, model_name, dataset_name, seqlen, topk, cache_path):
    """Pass 1: save base top-k log-probs + residual log-mass + argmax to disk."""
    print(f"[two_pass] Caching base top-{topk} log-probs for {dataset_name} -> {cache_path}")
    if seqlen == -1:
        seqlen = base_model.config.max_position_embeddings

    base_dev = _model_device(base_model)
    testenc = _get_testenc(model_name, dataset_name, seqlen)
    nsamples = testenc.numel() // seqlen

    batches = []
    for i in tqdm.tqdm(range(nsamples), desc=f"cache base {dataset_name}"):
        batch = testenc[:, i * seqlen : (i + 1) * seqlen].to(base_dev)
        logits = base_model(batch)["logits"].float()
        logp = F.log_softmax(logits, dim=-1)
        top_lp, top_idx = logp.topk(topk, dim=-1)
        top_mass = top_lp.exp().sum(-1).clamp(max=1.0 - 1e-6)
        resid_logmass = torch.log1p(-top_mass)
        argmax = logits.argmax(-1)
        batches.append({
            "top_idx": top_idx.to(torch.int32).cpu(),
            "top_lp": top_lp.to(torch.float16).cpu(),
            "resid_logmass": resid_logmass.to(torch.float16).cpu(),
            "argmax": argmax.to(torch.int32).cpu(),
        })
    torch.save({"seqlen": seqlen, "nsamples": nsamples, "topk": topk, "batches": batches}, cache_path)


@torch.no_grad()
def kl_eval_from_cache(quant_model, model_name, dataset_name, cache_path, log_wandb):
    """Pass 2: compute KL from cached base top-k plus residual-mass correction."""
    print(f"[two_pass] Evaluating KL on {dataset_name} from cache ...")
    cache = torch.load(cache_path, weights_only=False)
    seqlen = cache["seqlen"]
    nsamples = cache["nsamples"]

    quant_dev = _model_device(quant_model)
    testenc = _get_testenc(model_name, dataset_name, seqlen)

    kl_sum, top1_agree, total_tokens = 0.0, 0, 0
    per_seq_kl = []
    for i in tqdm.tqdm(range(nsamples), desc=f"KL {dataset_name} (pass 2)"):
        batch = testenc[:, i * seqlen : (i + 1) * seqlen].to(quant_dev)
        q_logits = quant_model(batch)["logits"].float()
        q_logp = F.log_softmax(q_logits, dim=-1)

        c = cache["batches"][i]
        top_idx = c["top_idx"].to(quant_dev).long()
        top_lp_b = c["top_lp"].to(quant_dev).float()
        resid_lp_b = c["resid_logmass"].to(quant_dev).float()
        base_argmax = c["argmax"].to(quant_dev).long()

        q_top_lp = q_logp.gather(-1, top_idx)
        base_top_p = top_lp_b.exp()
        kl_topk = (base_top_p * (top_lp_b - q_top_lp)).sum(-1)

        # Lump non-top-k mass as a single bin (llama.cpp-style approximation).
        q_top_mass = q_top_lp.exp().sum(-1).clamp(max=1.0 - 1e-6)
        q_resid_lp = torch.log1p(-q_top_mass)
        kl_resid = resid_lp_b.exp() * (resid_lp_b - q_resid_lp)

        kl_tok = kl_topk + kl_resid
        kl_sum += kl_tok.sum().item()
        top1_agree += (q_logits.argmax(-1) == base_argmax).sum().item()
        total_tokens += kl_tok.numel()
        per_seq_kl.append(kl_tok.mean().item())

    out = _reduce(kl_sum, top1_agree, total_tokens, per_seq_kl)
    print(f"{dataset_name} KL: {out}")
    if log_wandb:
        wandb.log({f"kl/{dataset_name}/{k}": v for k, v in out.items()})
    return out


def _save_result(results_path, quant_model_path, dataset, out, model_name, seqlen, mode, topk):
    path = (
        results_path if results_path != "None"
        else os.path.join(os.path.dirname(quant_model_path), "kl_results.yaml")
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            results = yaml.safe_load(f) or {}
    results[dataset] = {**out, "model_name": model_name, "seqlen": seqlen, "mode": mode, "topk": topk}
    with open(path, "w") as f:
        yaml.safe_dump(results, f)
    return path


def main():
    parser = argparse.ArgumentParser(description="KL divergence between base and quantized model")
    parser.add_argument("--model_name", type=str, required=True, help="HF model name (tokenizer + base weights)")
    parser.add_argument("--base_model_path", type=str, default=None, help="Override path for base model")
    parser.add_argument("--quant_model_path", type=str, required=True, help="Path to quantized/compressed checkpoint")
    parser.add_argument("--load_custom_model", action="store_true", help="Load quant via get_compressed_model_class")
    parser.add_argument("--dataset_names", type=str, nargs="+",
                        choices=["wikitext2", "c4"], default=["wikitext2", "c4"])
    parser.add_argument("--seqlen", type=int, default=-1)
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "dual_gpu", "two_pass"],
                        help="auto picks dual_gpu if both --base_gpu and --quant_gpu are given, else two_pass")
    parser.add_argument("--base_gpu", type=int, default=None, help="GPU index for base model (dual_gpu mode)")
    parser.add_argument("--quant_gpu", type=int, default=None, help="GPU index for quant model (dual_gpu mode)")
    parser.add_argument("--topk", type=int, default=128,
                        help="top-k retained in two_pass base cache (higher = closer to exact KL)")
    parser.add_argument("--kl_chunk_size", type=int, default=512,
                        help="Tokens per fp32 log-softmax chunk in dual_gpu mode (lower = less VRAM)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Where to write two_pass caches (default: system tempdir)")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--results_path", type=str, default="None")
    args = parser.parse_args()
    print("Arguments:", args)

    mode = args.mode
    if mode == "auto":
        mode = "dual_gpu" if (args.base_gpu is not None and args.quant_gpu is not None) else "two_pass"
    print(f"[kl_div] resolved mode: {mode}")

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

    results_path = None
    if mode == "dual_gpu":
        base_model = _load_base(args.model_name, args.base_model_path, gpu=args.base_gpu)
        quant_model = _load_quant(args.model_name, args.quant_model_path, args.load_custom_model, gpu=args.quant_gpu)
        base_model.eval(); quant_model.eval()
        for dataset in args.dataset_names:
            out = kl_eval_dual_gpu(base_model, quant_model, args.model_name, dataset, args.seqlen,
                                   args.log_wandb, chunk_size=args.kl_chunk_size)
            if args.save:
                results_path = _save_result(args.results_path, args.quant_model_path, dataset, out,
                                            args.model_name, args.seqlen, mode, topk=None)
    else:  # two_pass
        cache_dir = args.cache_dir or tempfile.mkdtemp(prefix="kl_cache_")
        os.makedirs(cache_dir, exist_ok=True)
        cache_paths = {ds: os.path.join(cache_dir, f"base_{ds}.pt") for ds in args.dataset_names}

        base_model = _load_base(args.model_name, args.base_model_path, gpu=args.base_gpu)
        base_model.eval()
        for dataset in args.dataset_names:
            cache_base_topk(base_model, args.model_name, dataset, args.seqlen, args.topk, cache_paths[dataset])
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

        quant_model = _load_quant(args.model_name, args.quant_model_path, args.load_custom_model, gpu=args.quant_gpu)
        quant_model.eval()
        for dataset in args.dataset_names:
            out = kl_eval_from_cache(quant_model, args.model_name, dataset, cache_paths[dataset], args.log_wandb)
            if args.save:
                results_path = _save_result(args.results_path, args.quant_model_path, dataset, out,
                                            args.model_name, args.seqlen, mode, topk=args.topk)
            try:
                os.remove(cache_paths[dataset])
            except OSError:
                pass

    print(f"Evaluation completed. Results saved to {results_path if args.save else 'not saved'}")


if __name__ == "__main__":
    main()
