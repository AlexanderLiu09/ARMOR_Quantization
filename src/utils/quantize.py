import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass

def quantize_per_group(x: torch.Tensor, n_bits: int, group_size: int, symmetric: bool =True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize a float tensor to integers using per-group symmetric quantization.
    
    Groups the values into contiguous chunks of `group_size`, computes one scale 
    per group as max(|values|) / q_max, and rounds each value to the nearest 
    integer on the grid [-q_max, +q_max].
    
    If the number of elements is not divisible by group_size, the tensor is 
    zero-padded to the next multiple before grouping, and the padding is 
    removed from the output.
    
    Args:
        x: Float tensor of any shape containing the values to quantize.
        n_bits: Bit width for quantization. Must be 4 or 8.
        group_size: Number of elements per quantization group. 
        symmetric: If True, use symmetric quantization (zero_point=None).
                   If False, use asymmetric (not implemented yet).
    
    Returns:
        x_int: Integer tensor with same shape as x, dtype torch.int8.
               Values are in [-q_max, q_max] for symmetric.
        scale: Float tensor of shape (num_groups,), same dtype as x.
               Each entry is the scale for one group: max(|group|) / q_max.
        zero_point: None for symmetric quantization.
    """
    original_shape = x.shape
     
    #Flatten to 1D
    x_flat, n = torch.flatten(x), x.numel()

    #Pad to integer multiple of group size
    pad_amount = 0
    if (n % group_size) != 0:
        pad_amount = group_size - (n % group_size)
        x_flat = torch.cat([x_flat, torch.zeros(pad_amount, device=x.device, dtype=x.dtype)], dim=0)

    #Reshape into (num groups x group size)
    x_grouped = x_flat.reshape(-1, group_size)

    if symmetric:
        q_max = 2**(n_bits -1) - 1
        max_abs = torch.amax(torch.abs(x_grouped), dim = 1, keepdim=True).clamp(min=1e-8)
        scale, zero_point = max_abs/q_max, None

        #Quantize
        x_int = torch.round(x_grouped/scale).clamp(-q_max, q_max).to(torch.int8)
    
    #Remove padding and reshape
    x_int = x_int.flatten()[:n].reshape(original_shape)
    scale = scale.squeeze(1)

    return (x_int, scale, zero_point)

def dequantize_per_group(x_int: torch.Tensor, scale: torch.Tensor, zero_point: Optional[torch.Tensor], group_size: int) -> torch.Tensor:
    """
    Dequantize an integer tensor back to float using per-group scales.
    
    Reverses the quantization performed by quantize_per_group(). Each group 
    of integer values is multiplied by its corresponding scale to recover 
    approximate float values.
    
    The padding/grouping logic must mirror quantize_per_group() exactly:
    if the original tensor was padded before quantization, this function 
    applies the same padding, performs the dequantization, then strips it.
    
    Args:
        x_int: Integer tensor (torch.int8) of any shape. Contains the 
               quantized values from quantize_per_group().
        scale: Float tensor of shape (num_groups,). One scale per group,
               as returned by quantize_per_group().
        zero_point: None for symmetric quantization. Reserved for future
                    asymmetric support.
        group_size: Number of elements per quantization group. Must match
                    the value used in quantize_per_group().
    
    Returns:
        x_float: Float tensor with same shape as x_int, same dtype as scale.
                 The dequantized approximation of the original values.
    """
    original_shape = x_int.shape

    #Flatten and cast int to float
    x_flat, n = x_int.flatten().to(scale.dtype), x_int.numel()
    
    #Pad to integer multiple of group size
    pad_amount = 0
    if (n % group_size) != 0:
        pad_amount = group_size - (n % group_size)
        x_flat = torch.cat([x_flat, torch.zeros(pad_amount, device = x_flat.device, dtype = x_flat.dtype)], dim = 0)

    #Reshape into (num groups x group size)
    x_grouped = x_flat.reshape(-1, group_size)

    if zero_point is None:
        x_float = x_grouped * scale.unsqueeze(1)

    #Remove padding and reshape
    x_float = x_float.flatten()[:n].reshape(original_shape)

    return x_float

class STEQuantize(torch.autograd.Function):
    """
    Straight-through estimator for quantization.
    
    Forward: quantize then dequantize (returns rounded float values)
    Backward: pass gradient through unchanged (as if rounding didn't happen)
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, n_bits: int, group_size: int, symmetric: bool) -> torch.Tensor:
        """
        Quantizes and then dequantizes x to obtain "rounded" value
        
        Args:
        x: Float tensor of any shape containing the values to quantize.
        n_bits: Bit width for quantization. Must be 4 or 8.
        group_size: Number of elements per quantization group. 
        symmetric: If True, use symmetric quantization (zero_point=None).
                   If False, use asymmetric (not implemented yet).
    
        Returns:
            x_deq: Dequantized x

        """  
        x_int, scale, zero_point = quantize_per_group(x, n_bits, group_size, symmetric)
        x_deq = dequantize_per_group(x_int, scale, zero_point, group_size)
        return x_deq

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None, None, None
    
def fake_quantize(x: torch.Tensor, n_bits: int = 8, group_size: int = 128, symmetric: bool = True) -> torch.Tensor:
    """
    Differentiable fake quantization using the straight-through estimator.
    
    In the forward pass, values are rounded to the nearest quantization grid 
    point (quantize then dequantize). In the backward pass, gradients pass 
    through unchanged as if the rounding didn't happen.
    
    This is the function called during training in the continuous update step.
    
    Args:
        x: Float tensor of any shape.
        n_bits: Bit width (4 or 8).
        group_size: Elements per quantization group.
        symmetric: Use symmetric quantization.
    
    Returns:
        Float tensor, same shape as x, with values snapped to the quantization grid.
    """

    return STEQuantize.apply(x, n_bits, group_size, symmetric)

@dataclass
class QuantConfig:
    """
    Configuration for quantization of the ARMOR sparse core.
    
    Passed through the training loop to control whether and how 
    fake quantization is applied during the continuous update step.
    
    Args:
        enabled: Whether quantization is active. When False, the entire 
                 system behaves identically to the original FP16 ARMOR.
        n_bits: Bit width for quantization (4 or 8).
        group_size: Number of nonzero values per quantization group.
        symmetric: Whether to use symmetric quantization (zero_point=0).
        qat_start_iter: Training iteration at which to begin fake quantization.
                        Iterations before this use pure FP16 (warmup phase).
    """
    enabled: bool = False
    n_bits: int = 8
    group_size: int = 128
    symmetric: bool = True
    qat_start_iter: int = 0




