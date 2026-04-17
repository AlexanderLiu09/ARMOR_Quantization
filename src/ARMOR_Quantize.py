import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
import os
import sys
import copy
import matplotlib.pyplot as plt
import itertools
import hydra
import tqdm
import random
import time
import wandb
from functools import partial # <-- Import partial
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Tuple, Optional, Union, List, Literal
from dataclasses import dataclass, field

from src.compression_parent import CompressedLinear
from src.sparse_compress import SparseLinear
from src.quantize_sparse import QuantizedSparseLinear
from src.utils import normalizer as normalize
from src.utils import utils
from src.utils.blockwise_diag_matricies import BlockwiseDiagMatrix
from src.ARMOR_compress import initalize_optimizer, TrainingConfig, SelectionConfig
from src.utils.quantize import fake_quantize_per_row, find_optimal_scale_per_row, QuantConfig, STEQuantizePerRow
    
class BlockCompressLearnable(nn.Module):
    original_weight: torch.FloatTensor
    importance_weight: Union[None, torch.FloatTensor] #shape of (d_in) if not None
    A: BlockwiseDiagMatrix #shape of (d_out, d_in)
    B: BlockwiseDiagMatrix #shape of (d_in, d_out)
    naive_compression_module: QuantizedSparseLinear
    block_size: int
    eps: float = 1e-8
    prev_dim: int = -1
    
    
    
    def __init__(self, original_weight: torch.FloatTensor, 
                 naive_compression_module: QuantizedSparseLinear,
                 block_size: Union[int, Tuple[int, int]],
                 importance_weight: Optional[torch.FloatTensor] = None,              
    ) -> None:
        """Initializes the PermutedSparseWeight class.

        Args:
            original_weight (torch.FloatTensor): The original weight matrix to be approximated.
            naive_compression_module (QuantizedSparseLinear): The naive compression module (e.g., SparseLinear).
            block_size (Union[int, Tuple[int, int]]): The size of the blocks for the block-diagonal matrices A and B. 
                If an integer is provided, both dimensions will use the same block size, otherwise a tuple specifying (block_size_A, block_size_B).
            importance_weight (Optional[torch.FloatTensor], optional): Importance weights for the input dimensions. Defaults to None.
        """
        
        super(BlockCompressLearnable, self).__init__()
        
        
        d_out, d_in = original_weight.shape
        
        if isinstance(block_size, int):
            block_size = (block_size, block_size)
            
            
        assert d_in % block_size[1] == 0, f"d_in = {d_in} must be divisible by block_size = {block_size[1]}"
        assert d_out % block_size[0] == 0, f"d_out = {d_out} must be divisible by block_size = {block_size[0]}"
        
        self.d_out = d_out
        self.d_in = d_in
            
        self.block_size = block_size
        
        #initalize each of the permutation matricies
        self.A = BlockwiseDiagMatrix(
            d = d_out,
            block_size=block_size[0],
            initalize_as_identity=True
        )
        self.B = BlockwiseDiagMatrix(
            d = d_in,
            block_size=block_size[1],
            initalize_as_identity=True
        )
                                                         
            
        self.original_weight = original_weight.detach().clone()
        self.original_weight.requires_grad = False
        
        self.importance_weight = importance_weight
        
        with torch.no_grad():
            self.loss_scaling = {"mean":1.0, "sum": 1.0}    
            self.loss_scaling = {"mean":self.recon_loss(reduction = "mean", zero_sub = True).item(),
                                    "sum": self.recon_loss(reduction = "sum", zero_sub = True).item()}
        
        self.naive_compression_module = naive_compression_module
        self.to(original_weight.device)
            

        
    #turn of gradient t
    def forward(self):
        S = self.naive_compression_module.reconstruct(method = "straight_through")

        return self.A @ S @ self.B            
        
    def recon_loss(self, reduction: Literal["mean", "sum", "none"] = "mean",
                   zero_sub:bool = False, recon_weight: Optional[torch.FloatTensor] = None,
                   **kwargs):
        if recon_weight is None:
            if zero_sub:
                recon_weight = torch.zeros_like(self.original_weight) #used to get the loss scaling
            else:
                recon_weight = self(**kwargs)
        
        #reconstruction loss
        recon_loss_elementwise = (recon_weight - self.original_weight) ** 2 if self.importance_weight is None else (recon_weight - self.original_weight) ** 2 * self.importance_weight.unsqueeze(0)

        if reduction == "mean":
            return recon_loss_elementwise.mean()/self.loss_scaling["mean"] #scale the loss by the scaling factor
        elif reduction == "sum":
            return recon_loss_elementwise.sum()/self.loss_scaling["sum"] #scale the loss by the scaling factor
        elif reduction == "none":
            return recon_loss_elementwise


    


@torch.no_grad()
def sparse_core_step(trainable_sparse: BlockCompressLearnable,
                     select: Literal["random", "gradient_greedy", "gradient_random"] = "random",
                     ) -> None:
    """Joint (mask, integer-quant) discrete update of the quantized sparse core.

    For each selected sparse group, exhaustively enumerates every (mask pattern,
    integer quant tuple) candidate and picks the combination with the lowest
    Hessian-weighted reconstruction cost. Uses the scales learned by the
    continuous STE optimizer as fixed inputs.

    Args:
        trainable_sparse (BlockCompressLearnable): The ARMOR decomposition module.
        n_times (int, optional): Number of discrete update sweeps. Defaults to 1.
        select: Group-selection strategy for choosing which sparse group to
            update per block. Only gradient-based selection is supported.
    """
    naive = trainable_sparse.naive_compression_module
    assert isinstance(naive, QuantizedSparseLinear), (
        f"sparse_core_step expects QuantizedSparseLinear, got {type(naive).__name__}."
    )

    # ---- sizing / constants ----
    block_size_0, block_size_1 = trainable_sparse.block_size
    d_out, d_in = trainable_sparse.original_weight.shape
    n_blocks_0 = d_out // block_size_0
    n_blocks_1 = d_in // block_size_1
    n_blocks_total = n_blocks_0 * n_blocks_1
    
    # print("Shapes")
    # print(f"d_out: {d_out}, d_in: {d_in}")
    # print(f"n_blocks_total: {n_blocks_total}, n_blocks_0: {n_blocks_0}, n_blocks_1: {n_blocks_1}")
    

    group_size = naive.sparse_group           # 4 for 2:4
    n_nonzero = naive.n_non_zero_per_group    # 2 for 2:4
    assert d_in % group_size == 0, (
        f"d_in={d_in} must be divisible by sparse group size {group_size}"
    )
    assert block_size_1 % group_size == 0, (
        f"block_size_1={block_size_1} must be divisible by sparse group size {group_size}"
    )
    n_groups_per_block_row = block_size_1 // group_size

    # quantization groupsize (e.g. 128) — each sparse 4-group lies entirely
    # inside one quant group, so a single scale applies across the whole group.
    groupsize = d_in // naive.n_scales_per_row
    assert groupsize % group_size == 0, (
        f"quant groupsize {groupsize} must be a multiple of sparse group size {group_size}"
    )

    device = trainable_sparse.original_weight.device
    dtype = trainable_sparse.A.diag_blocks.dtype

    # ---- fold sqrt(H) (importance) into B_diag (no inner normalizer to fold) ----
    A_diag = trainable_sparse.A.diag_blocks
    B_diag = trainable_sparse.B.diag_blocks
    if trainable_sparse.importance_weight is not None:
        B_diag = B_diag * torch.sqrt(
            trainable_sparse.importance_weight.view(n_blocks_1, 1, block_size_1)
        )

    # ---- pattern enumeration (C(group_size, n_nonzero) masks) ----
    possible_non_zero_idxs = torch.combinations(
        torch.arange(group_size, device=device), r=n_nonzero
    )  # (n_possible, n_nonzero)
    n_possible = possible_non_zero_idxs.shape[0]

    # ---- quantization range and number of floor/ceil combos ----
    q_min, q_max = naive.quant_range
    # For each pattern, each nonzero weight has 2 candidates (floor, ceil of wm*/scale).
    # With n_nonzero=2 this gives 2^2=4 combos per pattern → 6*4=24 total candidates.
    n_combos = 2 ** n_nonzero
    # Combo selector: (n_combos, n_nonzero) binary flags; bit i of c → 0=floor, 1=ceil
    combo_flags = torch.tensor(
        [[int((c >> i) & 1) for i in range(n_nonzero)] for c in range(n_combos)],
        device=device, dtype=dtype,
    )  # (n_combos, n_nonzero)

    selection_config = SelectionConfig.from_name(select)
    col_range = torch.arange(group_size, device=device)

    
    idxs = torch.arange(n_blocks_total, device=device)
    j = idxs // n_blocks_1  # block index along dim 0 (row-blocks)
    k = idxs % n_blocks_1   # block index along dim 1 (col-blocks)

    # ---- group selection: one sparse group per block ----
    if selection_config.random:
        raise NotImplementedError("Random selection not implemented for quantized sparse core step.")
    if selection_config.masked:
        raise NotImplementedError("Gradient masked selection not implemented for quantized sparse core step.")

    with torch.enable_grad():
        S_detached = naive.reconstruct_().detach().clone()
        S_detached.requires_grad = True
        loss = trainable_sparse.recon_loss(
            reduction="mean",
            recon_weight=trainable_sparse.A @ S_detached @ trainable_sparse.B,
        )
        loss.backward()
        grad = S_detached.grad  # (d_out, d_in)

    grad_blocked = (
        grad.view(n_blocks_0, block_size_0, n_blocks_1, block_size_1)
            .transpose(1, 2)
            .reshape(n_blocks_total, block_size_0 * n_groups_per_block_row, group_size)
    )
    if selection_config.norm == "L2":
        grad_norm = torch.norm(grad_blocked, p=2, dim=-1)
    elif selection_config.norm == "Linf":
        grad_norm = torch.norm(grad_blocked, p=float("inf"), dim=-1)
    elif selection_config.norm == "L1":
        grad_norm = torch.norm(grad_blocked, p=1, dim=-1)
    else:
        raise ValueError(f"Unsupported selection norm: {selection_config.norm}")

    if selection_config.greedy:
        _, selected_idxs = torch.max(grad_norm, dim=-1)
    else:
        selected_idxs = torch.multinomial(grad_norm + 1e-12, num_samples=1).squeeze(-1)

    idx_0 = j * block_size_0 + selected_idxs // n_groups_per_block_row
    idx_1 = k * block_size_1 + (selected_idxs % n_groups_per_block_row) * group_size
    group_idxs = torch.stack([idx_0, idx_1], dim=1)  # (n_blocks_total, 2)

    # ---- residual from zeroing out the selected groups ----
    original_naive = naive.reconstruct_().clone()
    original_naive[group_idxs[:, 0].unsqueeze(1),
                    group_idxs[:, 1].unsqueeze(1) + col_range.unsqueeze(0)] = 0.0

    W_remaining = trainable_sparse.A @ original_naive @ trainable_sparse.B - trainable_sparse.original_weight
    if trainable_sparse.importance_weight is not None:
        W_remaining = W_remaining * torch.sqrt(trainable_sparse.importance_weight.unsqueeze(0))
    W_remaining = (
        W_remaining.view(n_blocks_0, block_size_0, n_blocks_1, block_size_1)
                    .transpose(1, 2)
                    .reshape(n_blocks_total, block_size_0, block_size_1)
    )

    # ---- per-block a, B, H, g ----
    a = A_diag[j, :, idx_0 % block_size_0]  # (n_blocks_total, block_size_0)

    B = B_diag[k.unsqueeze(1),
                (group_idxs[:, 1] % block_size_1).unsqueeze(1) + col_range.unsqueeze(0),
                :]  # (n_blocks_total, group_size, block_size_1)

    # first-order term g = 2 * B @ W_rem^T @ a, shape (n_blocks_total, group_size)
    first_order_term = 2.0 * torch.bmm(
        torch.bmm(B, W_remaining.transpose(1, 2)),
        a.unsqueeze(2),
    ).squeeze(2)

    # ---- per-pattern H_p = B_p B_p^T and its inverse ----
    B_selected = B[:, possible_non_zero_idxs, :]  # (b, n_possible, n_nonzero, block_size_1)
    B_sq = torch.bmm(
        B_selected.reshape(-1, n_nonzero, block_size_1),
        B_selected.reshape(-1, n_nonzero, block_size_1).transpose(1, 2),
    ).view(n_blocks_total, n_possible, n_nonzero, n_nonzero)
    # Damped inverse via Cholesky (same as non-quantized ARMOR_compress.py)
    eye = torch.eye(n_nonzero, device=device, dtype=dtype).view(1, 1, n_nonzero, n_nonzero)
    B_sq_inv = torch.cholesky_inverse(
        torch.linalg.cholesky_ex(B_sq + eye * 1e-9)[0].reshape(-1, n_nonzero, n_nonzero)
    ).view(n_blocks_total, n_possible, n_nonzero, n_nonzero)

    g_all = first_order_term[:, possible_non_zero_idxs]  # (b, n_possible, n_nonzero)
    a_sq = (a ** 2).sum(dim=1)  # (n_blocks_total,)

    # ---- continuous LS optimum per pattern: wm*_p = -(H_p^{-1} @ (g_p/2)) / ||a||^2 ----
    rhs = (g_all / 2.0).unsqueeze(-1)  # (b, n_possible, n_nonzero, 1)
    wm_star = -torch.bmm(
        B_sq_inv.reshape(-1, n_nonzero, n_nonzero),
        rhs.reshape(-1, n_nonzero, 1),
    ).view(n_blocks_total, n_possible, n_nonzero) / (
        a_sq.view(-1, 1, 1) + trainable_sparse.eps
    )  # (b, n_possible, n_nonzero)

    # ---- per-block scales ----
    scale_idx = idx_1 // groupsize  # (n_blocks_total,)
    scales_for_blocks = naive.scales[idx_0, scale_idx, 0]  # (n_blocks_total,)
    # print(f"groupize: {groupsize}, scale_idx numel: {scale_idx.numel()}, scale shape: {naive.scales.shape}")
    scale = scales_for_blocks.view(-1, 1, 1)  # (b, 1, 1) for broadcasting

    # ---- build floor/ceil candidates: 2^n_nonzero combos per pattern ----
    # q_lo/q_hi: integer codes bracketing the continuous optimum, clamped to quant range
    wm_over_s = wm_star / scale                               # (b, n_possible, n_nonzero)
    q_lo = torch.floor(wm_over_s).clamp(q_min, q_max)        # (b, n_possible, n_nonzero)
    q_hi = torch.ceil(wm_over_s).clamp(q_min, q_max)         # (b, n_possible, n_nonzero)

    # q_combos[b, p, c, i]: integer code for entry i, combo c, pattern p, block b
    # = q_lo[b,p,i] + combo_flags[c,i] * (q_hi[b,p,i] - q_lo[b,p,i])
    q_delta = (q_hi - q_lo).unsqueeze(-2)  # (b, n_possible, 1, n_nonzero)
    q_combos = q_lo.unsqueeze(-2) + combo_flags.view(1, 1, n_combos, n_nonzero) * q_delta
    # shape (b, n_possible, n_combos, n_nonzero) = (b, 6, 4, 2) for 2:4 INT4

    # ---- evaluate cost(p, combo) = c0 * q^T H_p q + c1 * g_p^T q ----
    # c0 = a_sq * scale^2,  c1 = scale  (since s = q * scale)
    c0 = (a_sq * scales_for_blocks ** 2).view(-1, 1, 1)  # (b, 1, 1, 1)
    c1 = scales_for_blocks.view(-1, 1, 1)                # (b, 1, 1, 1)
    # H_p @ q_combo: einsum over last dim j of B_sq and q_combos
    Hq  = torch.einsum('bpij,bpcj->bpci', B_sq, q_combos)  # (b, 6, 4, 2)
    qHq = (q_combos * Hq).sum(dim=-1)                       # (b, 6, 4)
    gq  = (g_all.unsqueeze(-2) * q_combos).sum(dim=-1)      # (b, 6, 4)
    # print(f"gq shape: {gq.shape}, expected (n_blocks_total, n_possible, n_combos) = ({n_blocks_total}, 6, 4)")
    # print(f"qHq shape: {qHq.shape}, expected (n_blocks_total, n_possible, n_combos) = ({n_blocks_total}, 6, 4)")
    # print(f"c0 shape: {c0.shape}, expected (n_blocks_total, 1, 1, 1) = ({n_blocks_total}, 1, 1, 1)")
    # print(f"c1 shape: {c1.shape}, expected (n_blocks_total, 1, 1, 1) = ({n_blocks_total}, 1, 1, 1)")
    # print(f"first multiplication shape: {(c0 * qHq).shape}, expected (n_blocks_total, n_possible, n_combos) = ({n_blocks_total}, 6, 4)")
    # print(f"second multiplication shape: {(c1 * gq).shape}, expected (n_blocks_total, n_possible, n_combos) = ({n_blocks_total}, 6, 4)")
    cost = c0 * qHq + c1 * gq                               # (b, 6, 4)

    # ---- argmin over 6 * 4 = 24 candidates ----
    # print(f"cost shape: {cost.shape}, expected (n_blocks_total, n_possible, n_combos) = ({n_blocks_total}, 6, 4)")
    cost_flat = cost.reshape(n_blocks_total, n_possible * n_combos)  # (b, 24) #NOTE: this line is causing an error
    best_flat_idx = cost_flat.argmin(dim=1)                           # (b,)
    optimal_pattern_idx = best_flat_idx // n_combos                   # (b,) in [0, 6)
    optimal_combo_idx   = best_flat_idx %  n_combos                   # (b,) in [0, 4)

    optimal_non_zero_idxs = possible_non_zero_idxs[optimal_pattern_idx]  # (b, n_nonzero)
    b_idx = torch.arange(n_blocks_total, device=device)
    optimal_q = q_combos[b_idx, optimal_pattern_idx, optimal_combo_idx]  # (b, n_nonzero)
    sparse_values = optimal_q * scales_for_blocks.unsqueeze(1)           # (b, n_nonzero)

    # ---- write updated mask (clear the 4-group, set the chosen nonzeros) ----
    naive.sparse_mask[group_idxs[:, 0].unsqueeze(1),
                        group_idxs[:, 1].unsqueeze(1) + col_range.unsqueeze(0)] = False
    naive.sparse_mask[group_idxs[:, 0].unsqueeze(1),
                        optimal_non_zero_idxs + group_idxs[:, 1].unsqueeze(1)] = True
    naive.check_mask()

    # ---- write XQ (fake-quantized storage = q * scale; STE forward stays on-grid) ----
    naive.XQ.data[group_idxs[:, 0].unsqueeze(1),
                    group_idxs[:, 1].unsqueeze(1) + col_range.unsqueeze(0)] = 0.0
    naive.XQ.data[group_idxs[:, 0].unsqueeze(1),
                    optimal_non_zero_idxs + group_idxs[:, 1].unsqueeze(1)] = sparse_values
                
        
        
def loss_wrapper(trainable_sparse: BlockCompressLearnable):
    """
    for torch.compile()
    """        
        
    S =  trainable_sparse.naive_compression_module.reconstruct_(method="straight_through")
    residual = trainable_sparse.original_weight - trainable_sparse.A @ S @ trainable_sparse.B

    if trainable_sparse.importance_weight is not None:
        residual = residual * torch.sqrt(trainable_sparse.importance_weight.unsqueeze(0))
    
    return (residual * residual).mean() / trainable_sparse.loss_scaling["mean"]

        
compiled_loss = torch.compile(loss_wrapper, mode="default", dynamic=False)       


def initialize_optimizer(
    trainable_sparse: BlockCompressLearnable,
    optimizer_config: DictConfig,
    type: str):
    
    
    # #create the optimizers
    # trainable_sparse.A.init_optimizers(optimizer_config)
    # trainable_sparse.B.init_optimizers(optimizer_config)
    
    # params = []
    # #get all the parameters that are not in trainable_sparse.A and trainable_sparse.B
    # for name, param in trainable_sparse.named_parameters():
    #     if name.startswith("A.") or name.startswith("B."):
    #         continue
    #     if param.requires_grad:
    #         params.append(param)


    if type == "wrapper":
        optimizer = instantiate(
            optimizer_config, 
            params=list(trainable_sparse.A.parameters()) + list(trainable_sparse.B.parameters())
        )
    elif type == "core":
        optimizer = optimizer = instantiate(
            optimizer_config, 
            params=trainable_sparse.naive_compression_module.parameters()
        )
    else:
        raise ValueError("Incorrect optimizer type")
    
    return optimizer

def get_divisors(x):
    divisors = []
    for i in range(1, int(x**0.5) + 1):
        if x % i == 0:
            divisors.append(i)
            if i != x // i:  # Avoid adding the square root twice for perfect squares
                divisors.append(x // i)
    return sorted(divisors)

class QuantizedARMOR_Linear(CompressedLinear):
    name = "QuantizedARMOR_Linear"

    
    def ARMOR_sparse_(
        self,
        naive_compression_config: DictConfig,
        block_diagonal_config: DictConfig,
        optimizer_config: DictConfig,
        training_config: DictConfig,
        normalizer: Optional[normalize.Normalizer] = None,
        normalizer_kwargs: Optional[dict] = None,
        training_config_overrides: Optional[dict] = {},
    ):
        
        torch.set_num_threads(1)
        
        training_config:TrainingConfig = instantiate(training_config,
                                      **training_config_overrides) if training_config_overrides is not None else \
            instantiate(training_config)
        
        if training_config.logfile is not None:
            os.makedirs(os.path.dirname(training_config.logfile), exist_ok=True)
        

        normalized_weight = self.initialize_normalizer(
            normalizer=normalizer, normalizer_kwargs=normalizer_kwargs
        )
        
        #naive compression config consits of both the init config and the compression config
        self.naive_compression_module = instantiate(
            naive_compression_config.init_config,
            weight=normalized_weight,
            verbose=self.verbose,
        )

        self.naive_compression_module.hessianDiag = self.get_hessianDiag()
        
        #compress the weight with the naive compression module
        original_sparse = self.naive_compression_module.compress(
            **naive_compression_config.compression_config)
        #clean the naive compression module
        self.naive_compression_module.clean()
         
        #remove loss_weighting from the config
        del block_diagonal_config.importance_weight
        print("block diagonal config block size:", block_diagonal_config.block_size)
        trainable_sparse = BlockCompressLearnable(
            normalized_weight,
            naive_compression_module=self.naive_compression_module,
            importance_weight=self.get_hessianDiag(),
            # block_size=training_config.block_size_start,
            **block_diagonal_config,
        )
        #create the optimizers
        optimizer = initalize_optimizer(
            trainable_sparse=trainable_sparse,
            optimizer_config=optimizer_config,
        )
            
        start_time = time.time()
        #create a simple logger
        with torch.no_grad():
            # trainable_sparse()
            pre_quant_loss = trainable_sparse.recon_loss(reduction="mean", recon_weight = original_sparse).item()
            prev_iter_loss = compiled_loss(trainable_sparse).item() #trainable_sparse.recon_loss(reduction="mean").item()
            if self.verbose:
                print("Pre-quantization loss (using original sparse reconstruction):", pre_quant_loss)
                print(f"Initial loss: {prev_iter_loss}")
            if self.use_wandb:
                self.wandb_queue.put({"self.metric_name": pre_quant_loss,
                            self.step_metric: 0})
                self.wandb_queue.put({self.metric_name: prev_iter_loss,
                           self.step_metric: 1})
            # best_state_dict = copy.deepcopy(trainable_sparse.state_dict())

        #warmpup    
        with torch.no_grad():
            _ = compiled_loss(trainable_sparse)

        remaining_patience = training_config.overall_patience
        for i in tqdm.tqdm(range(training_config.n_iters), disable = not self.verbose):
            
            #CONTINOUS STEP
            #reset the optimizers
            optimizer.zero_grad()   
            
            recon_loss = compiled_loss(trainable_sparse) #trainable_sparse.recon_loss(reduction="mean") 
            loss = recon_loss
            
            loss.backward()
            #step the optimizers
            optimizer.step()
                    
            #SPARSE CORE STEP
            with torch.no_grad():
                sparse_core_step(
                    trainable_sparse,
                    select=training_config.sparse_core_step_select,
                )
                    
             #loss stuff
            with torch.no_grad():
                current_loss = compiled_loss(trainable_sparse).item() #trainable_sparse.recon_loss(reduction="mean").item()
                if current_loss > (1 - training_config.loss_rtol) * prev_iter_loss or current_loss > prev_iter_loss - training_config.loss_atol:
                    remaining_patience -= 1

                    if remaining_patience == 0:
                        if self.verbose:
                            print("Loss converged, stopping early")
                        break
                else:
                    remaining_patience = training_config.overall_patience
                    # best_state_dict = copy.deepcopy(trainable_sparse.state_dict())
                prev_iter_loss = current_loss
            # )
                
            if i%training_config.log_freq == training_config.log_freq-1 or i==0:
                log_str = f"Iter: {i}, Loss: {current_loss}"
                if self.verbose:
                    print(log_str)
                if self.use_wandb:
                    log = {self.metric_name: current_loss,
                           self.step_metric: i+1}
                    if self.direct_wandb_log:
                        wandb.log(log)
                    else:
                        #put the log in the queue
                        self.wandb_queue.put(log)
                if training_config.logfile is not None:
                    with open(training_config.logfile, "a") as f:
                        f.write(log_str + "\n")
                        
                    
            if training_config.iter_ablation:
                if i % training_config.iter_save_freq == training_config.iter_save_freq - 1  or (i==0 or i==training_config.n_iters-1):
                    print("saving iter", i)
                    #save the first iter and last iter and every iter_save_freq iterations
                    state_dict= {"A": trainable_sparse.A.state_dict(),
                                    "B": trainable_sparse.B.state_dict(),
                                    "naive_compression_module": trainable_sparse.naive_compression_module.state_dict()}
                    save_path = training_config.iter_save_path.replace("{iter}", str(i))
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    print("save_path", save_path)
                    torch.save(state_dict, save_path)
                    
            
        # trainable_sparse.load_state_dict(best_state_dict)
        
        if self.verbose:
            print("time taken to train:", time.time() - start_time)
            print("final loss:", trainable_sparse.recon_loss(reduction="mean").item())
            print("Finished training") 
            
        


        self.A = trainable_sparse.A
        self.B = trainable_sparse.B
        
            
        del trainable_sparse.naive_compression_module

    def load_iter_state(self, state_dict_path: str):
        state_dict = torch.load(state_dict_path, map_location=self.original_weight.device)
        self.A.load_state_dict(state_dict["A"])
        self.B.load_state_dict(state_dict["B"])
        # self.naive_compression_module.uncompress_sparse_values()
        self.naive_compression_module.load_state_dict(state_dict["naive_compression_module"])
        # self.naive_compression_module.compress_sparse_values()
        self.compressed = True
        
    def compress(self,
               naive_compression_config: DictConfig,
        block_diagonal_config: DictConfig,
        optimizer_config: DictConfig,
        training_config: DictConfig,
        normalizer: Optional[normalize.Normalizer] = None,
        normalizer_kwargs: Optional[dict] = None,
        training_config_overrides: Optional[dict] = {},
    ):
        self.compressed = True
        return self.ARMOR_sparse_(
            naive_compression_config = naive_compression_config,
            block_diagonal_config = block_diagonal_config,
            optimizer_config = optimizer_config,
            training_config = training_config,
            normalizer = normalizer,
            normalizer_kwargs = normalizer_kwargs,
            training_config_overrides = training_config_overrides,
        )
    
        
    
    def _no_checkpoint_forward(self, x: torch.FloatTensor):
        if self.forward_method == "reconstruct":
            if self.denormalization_method == "otf":
                y = F.linear(
                    self.normalizer.denormalize_otf_in(x),
                    self.reconstruct(denormalize=False),
                )
                y = self.normalizer.denormalize_otf_out(y) + (
                    self.bias if self.bias is not None else 0
                )
            else:
                # tqdm.tqdm.write(f"x dtype {x.dtype}, denormalize dtype {self.reconstruct(denormalize = self.denormalization_method == 'reconstruct').dtype}")
                y = F.linear(
                    x,
                    self.reconstruct(
                        denormalize=self.denormalization_method == "reconstruct"
                    ),
                    self.bias,
                )
        else:
            assert (
                self.denormalization_method == "otf"
            ), "on the fly denormalization is only supported for on the fly sparsity"
            
            x = self.normalizer.denormalize_otf_in(x)
            #multiply by B
            y = self.B(x.transpose(-2, -1)).transpose(-2, -1)
            #apply through the naive compression module
            y = self.naive_compression_module(y)
            #apply through A
            y = self.A(y.transpose(-2, -1)).transpose(-2, -1) #shape of (batch_size, d_out, d_in) 
            #denormalize the output
            y = self.normalizer.denormalize_otf_out(y)

        return y
    
    def reconstruct_(self, denormalize: bool = True) -> torch.FloatTensor:
        #reconstruct the weight
        weight = self.A(self.B(self.naive_compression_module.reconstruct(), leading=False), leading=True)
        

        if denormalize:
            weight = self.normalizer.denormalize(weight)
            

            
        return weight
    
    def blank_recreate(self,
                       block_diagonal_config: DictConfig,
                        normalizer: Optional[normalize.Normalizer] = None,
                        normalizer_kwargs: Optional[dict] = None,
                        naive_compression_config: DictConfig = None,
                        **kwargs
                            ):
        

        # print(kwargs)
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = normalize.Normalizer.blank_recreate(
                self.original_weight, **normalizer_kwargs
            )

        # print(kwargs["training_config"])
        training_config = instantiate(kwargs["training_config"])

        
        self.naive_compression_module = utils.blank_init(
            naive_compression_config,
            n_in = self.original_weight.shape[1],
            n_out = self.original_weight.shape[0],
            dtype=self.original_weight.dtype,
            device=self.original_weight.device,)

        if isinstance(block_diagonal_config.block_size, int):
            block_diagonal_config.block_size = (block_diagonal_config.block_size, block_diagonal_config.block_size)
        self.A = BlockwiseDiagMatrix(
            d=self.original_weight.shape[0],
            block_size=block_diagonal_config.block_size[0],
            initalize_as_identity=True,
            device=self.original_weight.device,
        )
        self.B = BlockwiseDiagMatrix(
            d=self.original_weight.shape[1],
            block_size=block_diagonal_config.block_size[1],
            initalize_as_identity=True,
            device=self.original_weight.device,
        )
        
        
            
            
        self.to(self.original_weight.device)
        
    
        
        self.compressed = True
        
    def get_additional_bits(self):
        
        additional_bits = self.A.get_n_bits() + self.B.get_n_bits()
        
        return additional_bits
    
    def get_n_bits(self):
        n_bits = self.naive_compression_module.get_n_bits() 
        n_bits += self.get_additional_bits() #no need to consider the normalizer as we can fold it into A and B
        return n_bits
    
    def get_n_nonzero(self):
        assert hasattr(self.naive_compression_module, "get_n_nonzero"), "Naive compression module does not have get_n_nonzero method"
        n_nonzero = self.naive_compression_module.get_n_nonzero()
        n_nonzero += (self.get_additional_bits())//16
        return n_nonzero
        
    @property
    def compression_measure(self):
        return self.naive_compression_module.compression_measure
        
        
        
#testing main fn 
if __name__ == "__main__":
    #@hydra.main(config_path="../config/compress", config_name="ARMOR")
    @hydra.main(config_path="../config/compress", config_name="ARMOR_quantized")
    def testing_main(cfg: DictConfig):
        utils.seed(0)
        device = "cuda:7"
        print("current_directory:", os.getcwd())
        model_name = "Qwen/Qwen3-8B-Base"
        # weight_path = "/data/lliu/NoWAG/models/meta-llama/Llama-2-7b-hf/original_weights/layer_28/mlp.up_proj.pt"
        proj_name = "layer_1/self_attn.q_proj"
        weight_path = f"../../../../LLM_data/{model_name}/original_weights/{proj_name}.pt"
        hessian_diag = weight_path.replace("original_weights", "hessian_diag/SlimPajama-627B/n_samples_128_ctx_len_8192/seed_0")

        
        weight = torch.load(weight_path, map_location=device)["weight"].to(torch.float32).detach()
        hessian_diag = torch.load(hessian_diag, map_location=device)["hessianDiag"].to(torch.float32 )
        
        #take the mean of all the finite valid hessian diag values
        with torch.no_grad():
            valid = torch.isfinite(hessian_diag) & (hessian_diag > 0)
            print("non-valid hessian diag values:", (~valid).sum().item(), "out of", hessian_diag.numel())
            mean_hessian_diag = hessian_diag[valid].mean().item()
            hessian_diag[~valid] = mean_hessian_diag
            hessian_diag = hessian_diag.detach()
        
        
        # assert torch.all(torch.isfinite(hessian_diag)), "Hessian diag contains non-finite or non-positive values"
        print("weight:", weight)
        print("hessian_diag:", hessian_diag)
        # weight = weight[:,:2048]
        # hessian_diag = hessian_diag[:2048]
        print("weight shape:", weight.shape)
        print("hessian_diag shape:", hessian_diag.shape)
        # raise ValueError("stop here")
        
        # print("created compression module gpu stats:")
        # print(utils.get_gpu_memory(weight.device))
        print("cfg:")
        #print out the cfg
        print(OmegaConf.to_yaml(cfg))
        # raise ValueError("stop here, we are done with training")
        compression_module = instantiate(
            cfg.init_config,
            weight = weight)
        compression_module.hessianDiag = hessian_diag
        # raise ValueError("stop here")
        torch.set_printoptions(linewidth = 240)
        compression_module.compress(
            training_config_overrides = {"iter_save_path": "test/test_run/permute_{iter}.pt",
                                         "logfile": "test/test_run/log.txt",
                                         "iter_save_freq": 1000},
            **cfg.compression_config
        )
        # torch.save({"A": compression_module.A.get_dense().to("cpu"),
        #             "B": compression_module.B.get_dense().to("cpu")},
        #         "/data/lliu/PermPrune/test/permute_compress_A_B.pt")
        # raise ValueError("stop here, we are done with training")
        
        
        torch.set_printoptions(sci_mode=False)
        
        #run some checks:
        print("reconstructued_weight:", compression_module.reconstruct(denormalize=True))
        print("original_weight:", weight)
        
        #create a random input 
        x = torch.randn(1, weight.shape[1]).to(device)
        
        #try several different forward pass methods
        y_naive = compression_module(x)
        
        
        compression_module.forward_method = "otf"
        compression_module.denormalization_method = "otf"
        y_otf = compression_module(x)
        print("y_naive:", y_naive)
        print("y_otf:", y_otf)
        print("maximum difference:", torch.max(torch.abs(y_naive - y_otf)))
        assert torch.allclose(y_naive, y_otf, atol=1e-5), "Naive and otf forward pass do not match"
        
        #try the blank recreate method
        
        state_dict = compression_module.state_dict()    
        
        new_compression_module = QuantizedARMOR_Linear(weight)
        
        new_compression_module.blank_recreate(
            **cfg.compression_config)
        print("state dict keys:", state_dict.keys())
        new_compression_module.load_state_dict(state_dict)
        
        assert torch.allclose(
            new_compression_module.reconstruct(), compression_module.reconstruct(), atol=1e-5
        ), "Weight does not match"
        
        y_blank_recreate = new_compression_module(x)
        assert torch.allclose(y_naive, y_blank_recreate, atol=1e-5), "Naive and blank recreate forward pass do not match"
        
        y_orig = F.linear(x, weight)
        print("y_orig:", y_orig)
        
        #test out load_iter_state 
        new_compression_module = QuantizedARMOR_Linear(weight)
        new_compression_module.blank_recreate(
            **cfg.compression_config)
        
        print(f"Path: test/test_run/permute_{cfg.compression_config.training_config.n_iters-1}.pt")
        new_compression_module.load_iter_state(
            f"test/test_run/permute_{cfg.compression_config.training_config.n_iters-1}.pt")
        
        new_compression_module.load_state_dict(state_dict)
        
        assert torch.allclose(
            new_compression_module.reconstruct(), compression_module.reconstruct(), atol=1e-5
        ), "Weight does not match"
        
        y_blank_recreate = new_compression_module(x)
        assert torch.allclose(y_naive, y_blank_recreate, atol=1e-5), "Naive and blank recreate forward pass do not match"
        
        y_orig = F.linear(x, weight)
        print("y_orig:", y_orig)
        
        if compression_module.compression_measure == "bits":
            print("average number  of bits", compression_module.get_n_bits()/compression_module.get_n_original_parameters())
        elif compression_module.compression_measure == "parameters":
            print("relative sparsity:", compression_module.get_n_nonzero()/compression_module.get_n_original_parameters())
        

    
    testing_main()