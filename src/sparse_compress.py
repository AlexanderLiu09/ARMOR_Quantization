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

class SparseCheckError(Exception):
    """Custom exception for sparse check errors."""
    pass

def get_group_and_n_nonzero(
    frac_nonzero: float,
    pattern: Optional[Tuple[int, int]] = None,
    sparse_group: Optional[int] = None,
    d_in: Optional[int] = None,
    d_out: Optional[int] = None,
)-> Tuple[int, int]:
    """Get the group size and number of non-zero elements per group based on the fraction of non-zero elements, pattern, and sparse group.
    """
    #get the sparse group size
    if pattern is not None:
        group = pattern[1]
        n_non_zero_per_group = pattern[0]
        #if this results in a different sparsity fraction, raise an error 
        if n_non_zero_per_group / group != frac_nonzero:
            print(f"Warning: sparse group size {group} and fraction {frac_nonzero} do not match, rounding up to {n_non_zero_per_group/group}")
    elif sparse_group is not None:
        group = sparse_group
        n_non_zero_per_group = math.ceil(
            group * frac_nonzero
        )
        if n_non_zero_per_group/ group != frac_nonzero:
            print(f"Warning: sparse group size {group} and fraction {frac_nonzero} do not match, rounding up to {n_non_zero_per_group/sparse_group}")
    else:
        assert frac_nonzero > 0, "frac_nonzero must be greater than 0"
        assert frac_nonzero <= 1, "frac_nonzero must be less than or equal to 1"
        group = d_in * d_out
        n_non_zero_per_group = math.ceil(
            group * frac_nonzero
        )
    return group, n_non_zero_per_group

class SparseLinear(compression_parent.CompressedLinear):
    name = "SparseLinear"
    compression_measure = "parameters"

    @torch.no_grad()
    def sparsify(
        self,
        frac_nonzero: float = 0.5,
        pattern: Optional[Tuple[int, int]] = None,
        sparse_group: Optional[Union[int, Literal["d_in"]]] = None,
        normalizer_kwargs: Optional[dict] = None,
        normalizer: Optional[normalize.Normalizer] = None,
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

        normalized_weight = self.initialize_normalizer(
            normalizer=normalizer, normalizer_kwargs=normalizer_kwargs
        )
        if isinstance(sparse_group, str) and sparse_group == "d_in":
            #if sparse_group is "d_in", set it to the input dimension
            sparse_group = self.in_features

        #get the sparse group size
        self.sparse_group, self.n_non_zero_per_group = get_group_and_n_nonzero(
            frac_nonzero=frac_nonzero,
            pattern=pattern,
            sparse_group=sparse_group,
            d_in=self.in_features,
            d_out=self.out_features,
        )
            
        #get the importances
        # print("normalized weight shape", normalized_weight.shape)
        # print("hessian diag shape", self.get_hessianDiag().shape)
        importances = normalized_weight**2 * self.get_hessianDiag().unsqueeze(0)
        importances = importances.view(-1, self.sparse_group)
        
        mask = torch.zeros(
            importances.shape, dtype=torch.bool, device=importances.device
        )
        
        #get the indices of the top k importances
        indices = torch.argsort(importances, dim=1, descending=True)[:, : self.n_non_zero_per_group]
        #set the mask to True for the top k importances
        mask.scatter_(1, indices, True)
        mask = mask.reshape_as(normalized_weight)
        
        
        self.X = nn.Parameter(normalized_weight[mask].detach().clone())
        self.sparse_mask = nn.Buffer(mask.detach().clone())
        
        if self.use_wandb:
            log = {
                    self.metric_name: self.get_reconstruction_error(
                        self.get_hessianDiag(), {"denormalize": False}
                    ).item(),
                    self.step_metric: 0
                }
            if self.direct_wandb_log:
                wandb.log(log)
            else:
                self.wandb_queue.put(log)
            
            
        
        
        
            

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
        return self.sparsify(
            frac_nonzero=frac_nonzero,
            pattern=pattern,
            sparse_group=sparse_group,
            normalizer_kwargs=normalizer_kwargs,
            normalizer=normalizer,
            **kwargs,
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
            raise DeprecationWarning(f"forward method {self.forward_method} not implemented")
        return y

    def get_n_bits(self):
        raise DeprecationWarning("get_n_bits is not no longer supported for SparseLinear, since it is not quantized. Use get_n_nonzero instead.")
        n_bits = 0
        if self.compressed:
            for sparse_module in self.sparse_modules:
                n_bits += sparse_module.get_n_bits()
        return n_bits

    def blank_recreate(
        self,
        frac_nonzero: float = 0.1,
        pattern: Optional[Tuple[int, int]] = None,
        sparse_group: Optional[int] = None,
        normalizer_kwargs: Optional[dict] = None,
        normalizer: Optional[normalize.Normalizer] = None,
        **kwargs,
    ):

        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = normalize.Normalizer.blank_recreate(
                self.original_weight, **normalizer_kwargs
            )
        if isinstance(sparse_group, str) and sparse_group == "d_in":
            #if sparse_group is "d_in", set it to the input dimension
            sparse_group = self.in_features
        self.sparse_group, self.n_non_zero_per_group = get_group_and_n_nonzero(
            frac_nonzero=frac_nonzero,
            pattern=pattern,
            sparse_group=sparse_group,
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
        
        self.X = nn.Parameter(
            torch.zeros(
                self.n_non_zero_per_group * self.get_n_original_parameters()// self.sparse_group,
                dtype=self.original_weight.dtype,
                device=self.original_weight.device,
            ))
        self.compressed = True
        # print("self.X.shape", self.X.shape)
        # print("original weight shape", self.original_weight.shape)

    def reconstruct_(self, denormalize: bool = True) -> torch.FloatTensor:
        """reconstructs the weigth matrix from the quantized version"""
        # print("reconstructing")
        if len(self.X.shape) == 1:
            reconstructed = torch.zeros((self.out_features, self.in_features), device=self.X.device, dtype=self.X.dtype)
            reconstructed[self.sparse_mask] = self.X
        else:
            reconstructed = self.X * self.sparse_mask

        if denormalize:
            reconstructed = self.normalizer.denormalize(reconstructed)

        return reconstructed

    def get_n_nonzero(self) -> int:

        recon = self.reconstruct(denormalize=False)

        n_nonzero = torch.sum(recon != 0).item()
        return n_nonzero

    @torch.no_grad()
    def compress_sparse_values(self):
        """Compress the sparse values to only keep the non-zero values"""
        assert len(self.X.shape) == 2, "X should be uncompressed first:"
        self.X = nn.Parameter(
            self.X[self.sparse_mask].detach().clone()
        )
    
    @torch.no_grad()
    def uncompress_sparse_values(self):
        """Uncompress the sparse values to their original shape"""
        self.X = nn.Parameter(
            self.reconstruct_(denormalize=False).detach().clone())
        
    def check_mask(self, 
                   mask: Optional[torch.Tensor] = None,
                   strict: bool = True,
                    verbose: bool = True) -> None:
        #check that the mask is correct
        if mask is None:
            mask = self.sparse_mask
        if not isinstance(mask, torch.Tensor):
            raise TypeError("mask should be a torch.Tensor")
        if mask.dtype != torch.bool:
            raise TypeError("mask should be a torch.bool tensor")
        mask_view = mask.view(-1, self.sparse_group)
        if strict:
            if not mask_view.sum(dim=1).eq(self.n_non_zero_per_group).all():
                if verbose:
                    mask_sum = mask_view.sum(dim=1)
                    print(f"mask is not correct, expected {self.n_non_zero_per_group} non-zero elements per group, got {mask_sum[mask_sum != self.n_non_zero_per_group]} non-zero elements")
                    #get the locations of the non-zero elements
                    non_zero_locations = torch.nonzero(mask_view.sum(dim=1) != self.n_non_zero_per_group)[:,0]
                    print(f"non-zero locations: {non_zero_locations}")
                    non_zero_idx_i = non_zero_locations * self.sparse_group // self.in_features
                    non_zero_idx_j = non_zero_locations * self.sparse_group % self.in_features
                    print(f"non-zero indices: {non_zero_idx_i}, {non_zero_idx_j}")
                    print(f"mask at non-zero indices: {self.sparse_mask[non_zero_idx_i, non_zero_idx_j:non_zero_idx_j + self.sparse_group]}")
                    print(f"mask is not correct, expected {self.n_non_zero_per_group} non-zero elements per group, got {mask_sum[mask_sum != self.n_non_zero_per_group]} non-zero elements")
                raise SparseCheckError
        else:
            #just check that the number of non-zero elements is LESS than the expected number
            if not mask_view.sum(dim=1).le(self.n_non_zero_per_group).all():
                if verbose:
                    # print(f"mask is not correct, expected at most {self.n_non_zero_per_group} non-zero elements per group, got {mask_view.sum(dim=1)[mask_view.sum(dim=1) > self.n_non_zero_per_group]} non-zero elements")
                    mask_sum = mask_view.sum(dim=1)
                    print(f"mask is not correct, expected at most {self.n_non_zero_per_group} non-zero elements per group, got {mask_sum[mask_sum > self.n_non_zero_per_group]} non-zero elements")
                    #get the locations of the non-zero elements
                    non_zero_locations = torch.nonzero(mask_view.sum(dim=1) > self.n_non_zero_per_group)[:,0]
                    print(f"non-zero locations: {non_zero_locations}")
                    non_zero_idx_i = non_zero_locations * self.sparse_group // self.in_features
                    non_zero_idx_j = non_zero_locations * self.sparse_group % self.in_features
                    print(f"non-zero indices: {non_zero_idx_i}, {non_zero_idx_j}")
                    print(f"mask at non-zero indices: {self.sparse_mask[non_zero_idx_i, non_zero_idx_j:non_zero_idx_j + self.sparse_group]}")
                    print(f"mask is not correct, expected at most {self.n_non_zero_per_group} non-zero elements per group, got {mask_sum[mask_sum > self.n_non_zero_per_group]} non-zero elements")
                raise SparseCheckError