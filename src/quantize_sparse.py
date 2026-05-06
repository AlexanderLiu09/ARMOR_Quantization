import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
import os
import tqdm
import wandb
from typing import Tuple, Optional, Union, List, Literal
import src.utils.normalizer as normalize
import src.compression_parent as compression_parent
from src.sparse_compress import SparseLinear, get_group_and_n_nonzero




class QuantizedSparseLinear(SparseLinear):
    name = "QuantizedSparseLinear"
    compression_measure = "bits"
    quantized = True 

    @torch.no_grad()
    def sparsify_and_quantize(
        self,
        pattern = (2, 4),
        quant_precision: int = 4,
        groupsize: Literal[-1, 128] = 128,
        **kwargs,
    ):
        """create a sparse compensator

        Args:
            frac_nonzero (float, optional): fraction of the weights to be nonzero. Defaults to 0.1.
            pattern (Tuple[int, int], optional): pattern of the sparsity for N:M sparsity where there are N nonzero 
            for every M. Defaults to None.
            sparse_group (int, optional): group of indices to be sparse. Defaults to None.
            normalizer_kwargs (Optional[dict], optional): kwargs for the normalizer. Defaults to None.
            normalizer (Optional[normalize.Normalizer], optional): normalizer to use. Defaults to None.
        """

        #check that the pattern is set to 2:4 to prevent insanity 
        print("pattern", pattern, type(pattern))
        assert pattern == (2, 4), "only 2:4 sparsity pattern is supported for quantized sparse linear for now"
        
        
        # for now we will only support 
        self.sparsify(
            pattern=pattern,
            normalizer_kwargs=None,
            normalizer=None,
        )
        
        #now quantize the sparse values 
        self.quant_precision = quant_precision
        self.quant_range = (-2 ** (quant_precision - 1), 2 ** (quant_precision - 1) - 1)
        XQ = self.reconstruct_(method = "super").detach()

        assert self.in_features % groupsize == 0, "in_features must be divisible by groupsize"
        self.n_scales_per_row = self.in_features // groupsize
        
        scales = XQ.view(self.out_features, self.n_scales_per_row, -1).abs().amax(dim=2, keepdim=True) / self.quant_range[1]
        XQ = torch.round(XQ.view(self.out_features, self.n_scales_per_row, -1) / scales).clamp(*self.quant_range) * scales
        
        #in essence keep a master copy of the quantized value in full precision, but we will 
        #only output the quantized value during reconstruction 
        self.XQ = nn.Parameter(XQ.view(self.out_features, self.in_features))
        self.scales = nn.Parameter(scales)
        #return a copy of the original sparse weight so we can compare the loss 
        NonQuantizedSparseOut = self.reconstruct_(method = "super").detach()
        del self.X

        return  NonQuantizedSparseOut
        
    def reconstruct_(self, method: Literal["super", "straight_through", "standard"] = "standard"):
        
        # if super, use the original reconstruct method, this will be only used during the compress
        if method == "super":
            assert hasattr(self, "X"), "X must be defined for super reconstruct"
            return super().reconstruct_(denormalize=False)

        # if standard 
        if method == "standard":
            with torch.no_grad():
                X_scaled = self.XQ.view(self.out_features, self.n_scales_per_row, -1) / self.scales
                X_rounded = torch.round(X_scaled).clamp(*self.quant_range)
                X_quantized = X_rounded * self.scales
                X_quantized = X_quantized.view(self.out_features, self.in_features)
                return X_quantized * self.sparse_mask
            
        if method == "straight_through":
            #this one will have a graidnet to both the quantized values and the scales
            #straight through estimator for quantization
            X_scaled = self.XQ.view(self.out_features, self.n_scales_per_row, -1) / self.scales
            
            X_scaled_ste = (torch.round(X_scaled) - X_scaled).detach() + X_scaled
            
            #clamp to quantization range
            X_scaled_ste = X_scaled_ste.clamp(*self.quant_range)
            X_quantized = X_scaled_ste * self.scales
            #reshape back to original shape
            X_quantized = X_quantized.view(self.out_features, self.in_features)
            return X_quantized * self.sparse_mask
            
            
            
        
            

    def compress(
        self,
        frac_nonzero: float = 0.1,
        pattern: Optional[Tuple[int, int]] = None,
        sparse_group: Optional[int] = None,
        normalizer_kwargs: Optional[dict] = None,
        normalizer: Optional[normalize.Normalizer] = None,
        **kwargs,
    ):
        self.compressed = True
        return self.sparsify_and_quantize(
            frac_nonzero=frac_nonzero,
            pattern=pattern,
            sparse_group=sparse_group,
            normalizer_kwargs=normalizer_kwargs,
            normalizer=normalizer,
            **kwargs,
        )

    def _no_checkpoint_forward(self, x: torch.FloatTensor):
        W = self.reconstruct_(method = "standard")
        y = F.linear(x, W, self.bias)
        
        return y

    def get_n_bits(self):
        n_bits = 0
        assert self.compressed, "model must be compressed to get n_bits"        
        
        
        n_bits = (self.quant_precision +  2) * torch.sum(self.sparse_mask).item()
        #+2 for the positioning bit overhead 
        n_bits = n_bits + self.out_features * self.n_scales_per_row * 16
        return n_bits
        
    def blank_recreate(
        self,
        pattern = (2, 4),
        quant_precision: int = 4,
        groupsize: Literal[-1, 128] = 128,
        **kwargs,
    ):
        self.sparse_group, self.n_non_zero_per_group = get_group_and_n_nonzero(
            frac_nonzero=0.5,
            pattern=pattern,
            sparse_group=None,
            d_in=self.in_features,
            d_out=self.out_features,
        )
        
        sparse_mask = torch.ones(self.original_weight.shape, dtype=torch.bool,
                          device=self.original_weight.device)
        sparse_mask = sparse_mask.view(-1, self.sparse_group)
        sparse_mask[:, self.n_non_zero_per_group:] = False
        sparse_mask = sparse_mask.view(self.original_weight.shape)
        self.sparse_mask = nn.Buffer(
            sparse_mask
        )
        self.n_scales_per_row = self.in_features // groupsize
        
        self.XQ = nn.Parameter(
            torch.zeros(
                (self.out_features, self.in_features),
                dtype=self.original_weight.dtype,
                device=self.original_weight.device,
            ))
        self.scales = nn.Parameter(
            torch.zeros(
                (self.out_features, self.n_scales_per_row, 1),
                dtype=self.original_weight.dtype,
                device=self.original_weight.device,
            )
        )
        self.quant_precision = quant_precision
        self.quant_range = (-2 ** (quant_precision - 1), 2 ** (quant_precision - 1) - 1)
        self.compressed = True
        
        
        # raise NotImplementedError("blank_recreate is not implemented for QuantizedSparseLinear, since it is not needed. Use compress instead.")
        # if normalizer is not None:
        #     self.normalizer = normalizer
        # else:
        #     self.normalizer = normalize.Normalizer.blank_recreate(
        #         self.original_weight, **normalizer_kwargs
        #     )
        # if isinstance(sparse_group, str) and sparse_group == "d_in":
        #     #if sparse_group is "d_in", set it to the input dimension
        #     sparse_group = self.in_features
        # self.sparse_group, self.n_non_zero_per_group = get_group_and_n_nonzero(
        #     frac_nonzero=frac_nonzero,
        #     pattern=pattern,
        #     sparse_group=sparse_group,
        #     d_in=self.in_features,
        #     d_out=self.out_features,
        # )
        
        # sparse_mask = torch.ones(self.original_weight.shape, dtype=torch.bool,
        #                   device=self.original_weight.device)
        # sparse_mask = sparse_mask.view(-1, self.sparse_group)
        # sparse_mask[:, self.n_non_zero_per_group:] = False
        # sparse_mask = sparse_mask.view(self.original_weight.shape)
        
        # self.sparse_mask = nn.Buffer(
        #     sparse_mask
        # )
        
        # self.X = nn.Parameter(
        #     torch.zeros(
        #         self.n_non_zero_per_group * self.get_n_original_parameters()// self.sparse_group,
        #         dtype=self.original_weight.dtype,
        #         device=self.original_weight.device,
        #     ))
        # self.compressed = True
        # # print("self.X.shape", self.X.shape)
        # # print("original weight shape", self.original_weight.shape)


    def get_n_nonzero(self) -> int:


        n_nonzero = self.sparse_mask.numel() - self.sparse_mask.sum().item()
        return n_nonzero



