import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass


@torch.no_grad()
def find_optimal_scale_per_row(x_rows: torch.Tensor, n_bits: int,
                                n_grid: int = 30,
                                hessian_weights: Optional[torch.Tensor] = None,
                                chunk_rows: int = 1024) -> torch.Tensor:
    """
    Grid search for optimal per-row symmetric quantization scale.

    For each row, tries n_grid candidate scales linearly spaced from
    s_max/n_grid to s_max (where s_max = max(|row|) / q_max) and selects
    the scale that minimizes (optionally Hessian-weighted) squared error.

    Args:
        x_rows: [d_out, nnz_per_row] values to quantize, one row per output channel.
        n_bits: Bit width for quantization (e.g. 4).
        n_grid: Number of candidate scales to evaluate.
        hessian_weights: Optional importance weights.
            - [nnz_per_row]: same weights broadcast across all rows.
            - [d_out, nnz_per_row]: per-row per-element weights.
        chunk_rows: Process this many rows at a time to limit memory.

    Returns:
        scale: [d_out] optimal scale per row.
    """
    q_max = 2 ** (n_bits - 1) - 1
    d_out = x_rows.shape[0]

    s_max = x_rows.abs().amax(dim=1).clamp(min=1e-8) / q_max          # [d_out]
    alphas = torch.linspace(1.0 / n_grid, 1.0, n_grid,
                            device=x_rows.device, dtype=x_rows.dtype)  # [n_grid]

    best_scale = torch.empty(d_out, device=x_rows.device, dtype=x_rows.dtype)

    for start in range(0, d_out, chunk_rows):
        end = min(start + chunk_rows, d_out)
        x_chunk = x_rows[start:end]                                    # [chunk, nnz]
        s_max_chunk = s_max[start:end]                                 # [chunk]

        s_cand = s_max_chunk[:, None] * alphas[None, :]                # [chunk, n_grid]

        x_exp = x_chunk[:, None, :]                                    # [chunk, 1, nnz]
        s_exp = s_cand[:, :, None]                                     # [chunk, n_grid, 1]

        x_deq = torch.clamp(torch.round(x_exp / s_exp), -q_max, q_max) * s_exp
        err_sq = (x_deq - x_exp) ** 2                                 # [chunk, n_grid, nnz]

        if hessian_weights is not None:
            if hessian_weights.dim() == 1:
                h = hessian_weights[None, None, :]                     # [1, 1, nnz]
            else:
                h = hessian_weights[start:end, None, :]                # [chunk, 1, nnz]
            err_sq = err_sq * h

        loss = err_sq.sum(dim=2)                                       # [chunk, n_grid]
        best_idx = loss.argmin(dim=1)                                  # [chunk]
        best_scale[start:end] = s_cand[torch.arange(end - start, device=s_cand.device), best_idx]

    return best_scale


def quantize_per_row(x_rows: torch.Tensor, n_bits: int,
                      scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D tensor using per-row symmetric scales.

    Args:
        x_rows: [d_out, nnz_per_row] float values.
        n_bits: Bit width.
        scale: [d_out] per-row scale factors.

    Returns:
        x_int: [d_out, nnz_per_row] as int8 (values in [-q_max, q_max]).
        scale: [d_out] (passed through for convenience).
    """
    q_max = 2 ** (n_bits - 1) - 1
    x_int = torch.round(x_rows / scale[:, None]).clamp(-q_max, q_max).to(torch.int8)
    return x_int, scale


def dequantize_per_row(x_int: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Dequantize using per-row scales.

    Args:
        x_int: [d_out, nnz_per_row] integer tensor.
        scale: [d_out] per-row scale factors.

    Returns:
        x_float: [d_out, nnz_per_row] dequantized float values.
    """
    return x_int.to(scale.dtype) * scale[:, None]


class STEQuantizePerRow(torch.autograd.Function):
    """
    Straight-through estimator for per-row symmetric quantization.

    Forward: quantize then dequantize using per-row scales.
    Backward: pass gradient through unchanged.
    """

    @staticmethod
    def forward(ctx, x_rows: torch.Tensor, n_bits: int, scale: torch.Tensor) -> torch.Tensor:
        q_max = 2 ** (n_bits - 1) - 1
        return torch.clamp(torch.round(x_rows / scale[:, None]), -q_max, q_max) * scale[:, None]

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def fake_quantize_per_row(x_rows: torch.Tensor, n_bits: int = 4,
                           n_grid: int = 100,
                           hessian_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Differentiable fake quantization with per-row Hessian-aware grid-searched scales.

    Finds the optimal per-row scale via grid search (minimizing Hessian-weighted
    squared error), then applies quantize-dequantize with straight-through
    gradient estimation.

    Args:
        x_rows: [d_out, nnz_per_row] float values.
        n_bits: Bit width.
        n_grid: Grid search resolution.
        hessian_weights: Optional [nnz_per_row] or [d_out, nnz_per_row] importance weights.

    Returns:
        [d_out, nnz_per_row] fake-quantized values (same shape as input).
    """
    scale = find_optimal_scale_per_row(x_rows, n_bits, n_grid, hessian_weights)
    return STEQuantizePerRow.apply(x_rows, n_bits, scale)


@dataclass
class QuantConfig:
    """
    Configuration for per-row symmetric quantization of the ARMOR sparse core.

    Uses grid search to find optimal per-row scales, optionally weighted
    by Hessian diagonal for activation-aware scale selection.
    """
    enabled: bool = False
    n_bits: int = 4
    n_grid: int = 100
    qat_start_iter: int = 0
