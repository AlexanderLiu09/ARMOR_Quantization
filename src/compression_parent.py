import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
import os
from typing import Tuple, Optional, Union, List, Literal
import src.utils.compress as compress_utils
from src.utils.normalizer import Normalizer
import src.utils.utils as utils
from typing import Optional, Union
from dataclasses import dataclass


@dataclass
class PlaceholderOrigWeight:
    shape: Tuple[int, int]
    device: Optional[Union[str, torch.device]] = None
    dtype: Optional[torch.dtype] = None


class CompressedLinear(nn.Module):
    """Parent class of all compression algorithms for linear layers"""

    name = "CompressedLinear"
    def __init__(
        self,
        weight: torch.FloatTensor,
        bias: Optional[torch.FloatTensor] = None,
        add_bias: bool = False,
        verbose: bool = False,
    ):
        """quantized linear layer

        Args:
            weight (torch.FloatTensor): the original weight matrix of shape (out_features, in_features)
            bias (Optional[torch.FloatTensor], optional): the original bias vector of shape (out_features) or None.
            add_bias (bool, optional): should we add a bias to the layer or not. Defaults to False.
        """

        super(CompressedLinear, self).__init__()
        self.out_features, self.in_features = weight.shape
        self.original_weight = weight
        # print("original weight", self.original_weight[0])
        # self.register_buffer("original_weight", weight)
        self.original_parameters = self.in_features * self.out_features

        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=True)
            self.original_parameters += self.out_features

        else:
            if add_bias:
                self.bias = nn.Parameter(
                    torch.zeros(self.out_features), requires_grad=True
                )
            else:
                self.bias = None

        self.compressed = False
        self.grad_checkpoint = False
        self.verbose = verbose
        self.denormalization_method: Literal["otf", "reconstruct", "ignore"] = (
            "reconstruct"
        )
        self.forward_method: Literal["reconstruct", "otf"] = "reconstruct"
        self.use_wandb = False
        
    def enable_wandb(self, metric_name: str = "loss",
                     step_metric: str = "step",
                    wandb_queue: Optional[torch.multiprocessing.Queue] = None):
        self.use_wandb = True
        self.metric_name = metric_name
        self.step_metric = step_metric
        if wandb_queue is not None:
            self.wandb_queue = wandb_queue
            self.direct_wandb_log = False
        else:
            self.direct_wandb_log = True
            # raise ValueError("wandb_queue cannot be None, please pass a valid queue")
        
        

    def compress(
        self,
        normalizer_kwargs: Optional[dict] = None,
        normalizer: Optional[Normalizer] = None,
        **kwargs,
    ):
        """compress the weights, this is the main function to be implemented by the child classes"""
        self.compressed = True
        raise NotImplementedError

    # helper function to initialize the normalizer
    def initialize_normalizer(
        self,
        normalizer_kwargs: Optional[dict] = None,
        normalizer: Optional[Normalizer] = None,
    ):
        """Two ways to initialize the normalizer, either pass the normalizer or the normalizer_kwargs

        Args:
            normalizer_kwargs (Optional[dict], optional): kwargs for normalizer. Defaults to None.
            normalizer (Optional[compress_parent.Normalizer], optional): normalizer class. Defaults to None.
        """
        if normalizer is not None:
            self.normalizer = normalizer
            normalized_weight = self.normalizer.normalize(self.original_weight)
        else:
            # print("normalizer_kwargs", normalizer_kwargs)
            if normalizer_kwargs is None:
                print("Warning: normalizer_kwargs is None, using default")
                normalizer_kwargs = {}
            self.normalizer, normalized_weight = Normalizer.normalize_init(
                self.original_weight, **normalizer_kwargs
            )

        return normalized_weight

    def reconstruct_(self, denormalize: bool = True) -> torch.FloatTensor:
        """reconstructs the weigth matrix from the compressed version"""
        raise NotImplementedError

    def _no_checkpoint_forward(self, x: torch.FloatTensor):
        raise NotImplementedError

    def blank_recreate(self, **kwargs):
        """recreates the compressed layer without any compression"""
        raise NotImplementedError

    def get_n_bits(self) -> int:
        """returns the number of bits needed to store the compressed layer"""
        raise NotImplementedError

    # ================ Another Initialization Fns =================
    @classmethod
    def blank_init(
        cls,
        n_in: int,
        n_out: int,
        add_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        compression_config: Optional[dict] = None,
        # **kwargs,
    ):
        """creates a blank layer with the same shape as the original layer"""
        # print("device", device,"dtype", dtype)
        placeholder_weight = PlaceholderOrigWeight(
            (n_out, n_in), device=device, dtype=dtype
        )
        layer = cls(placeholder_weight, 
                    add_bias=compression_config.get("add_bias", add_bias) if not add_bias else add_bias)
        #override the add bias if we need it, this is necessary for models such as Qwen2 which 
        #have a bias in the linear layer that we shouldn't remove
        layer.blank_recreate(**compression_config)
        return layer

    # ================= Importance/Logging Fns =================
    def enable_hessian_logging(
        self,
        hessian: Optional[torch.FloatTensor] = None,
        logging_type: Literal["mean", "ema"] = "mean",
        **kwargs,
    ):
        """enable hessian logging"""
        if hessian is not None:
            self.register_buffer("hessian", hessian)
        else:
            self.register_buffer(
                "hessian",
                torch.zeros(
                    self.in_features,
                    self.in_features,
                    device=self.original_weight.device,
                    dtype=torch.float32,
                ),
            )
        if logging_type == "mean":
            self.n_samples = kwargs.get("n_samples", 0)  # allows us to continue logging
            self.hessian_handle = self.register_forward_pre_hook(
                compress_utils.hessian_mean_logging
            )
        elif logging_type == "ema":
            self.decay = kwargs.get("decay", 0.99)
            self.hessian_handle = self.register_forward_pre_hook(
                compress_utils.hessian_ema_logging
            )
        else:
            raise ValueError(f"logging_type {logging_type} not supported")

    def enable_hessianDiag_logging(
        self,
        hessianDiag: Optional[torch.FloatTensor] = None,
        logging_type: Literal["mean", "ema"] = "mean",
        **kwargs,
    ):
        """enable hessianDiag logging
        hessianDiag are just the diagonal of the hessian
        """
        if hessianDiag is not None:
            self.register_buffer("hessianDiag", hessianDiag)
        else:
            self.register_buffer(
                "hessianDiag",
                torch.zeros(
                    self.in_features,
                    device=self.original_weight.device,
                    dtype=torch.float32,
                ),
            )

        if logging_type == "mean":
            self.n_samples = kwargs.get("n_samples", 0)
            self.hessianDiag_handle = self.register_forward_pre_hook(
                compress_utils.hessianDiag_mean_logging
            )
        elif logging_type == "ema":
            self.decay = kwargs.get("decay", 0.99)
            self.hessianDiag_handle = self.register_forward_pre_hook(
                compress_utils.hessianDiag_ema_logging
            )

    def dump_hessian(self) -> List[torch.FloatTensor]:
        """gives the hessian calculated and stops logging the inputs for the hessian

        Returns:
            torch.FloatTensor: the hessian
        """
        hessian = self.hessian.clone()
        self.hessian_handle.remove()
        del self.hessian_handle
        del self.hessian
        del self.n_samples
        return [hessian]  # returning a list for consistency with the low rank sparse

    def get_hessian(self) -> torch.FloatTensor:
        if hasattr(self, "hessian"):
            return self.hessian
        else:
            raise Exception("No hessian found")

    def get_hessianDiag(self) -> torch.FloatTensor:
        if hasattr(self, "hessianDiag"):  # new format that saves space
            hessianDiag = self.hessianDiag
        elif hasattr(self, "hessian"):  # old format
            hessianDiag = torch.diag(self.hessian)
        else:
            raise Exception("No hessian found")
        return hessianDiag

    def dump_hessianDiag(self) -> List[torch.FloatTensor]:
        """gives the importances calculated and stops logging the inputs for the importances

        Returns:
            torch.FloatTensor: the importances
        """
        hessianDiag = self.hessianDiag.clone()
        if hasattr(self, "hessianDiag_handle"):
            self.hessianDiag_handle.remove()
            del self.hessianDiag_handle
            del self.hessianDiag
            del self.n_samples
        return [
            hessianDiag
        ]  # returning a list for consistency with the low rank sparse

    @torch.no_grad()
    def validate_hessianDiag(self):
        
        with torch.no_grad():
            valid = torch.isfinite(self.hessianDiag) & (self.hessianDiag > 0)
            if torch.sum(~valid) > 0:
                print("non-valid hessian diag values:", (~valid).sum().item(), "out of", self.hessianDiag.numel())
                mean_hessian_diag = self.hessianDiag[valid].mean().item()
                self.hessianDiag[~valid] = mean_hessian_diag
                self.hessianDiag = self.hessianDiag.detach()
        
    # ================= Forward Fns =================
    def forward(self, x: torch.FloatTensor):
        """forward pass of the linear layer"""
        if not self.compressed:
            return F.linear(x, self.original_weight, self.bias)
        if self.grad_checkpoint:
            return self._checkpoint_forward(x)
        else:
            return self._no_checkpoint_forward(x)

    def _checkpoint_forward(self, x: torch.FloatTensor):
        return torch.utils.checkpoint.checkpoint(
            self._no_checkpoint_forward, x, use_reentrant=True
        )

    # ================= Reconstruction Fns =================
    def reconstruct(self, **kwargs) -> torch.FloatTensor:
        """reconstructs the weigth matrix from the quantized version"""
        if hasattr(self, "cached_reconstruct"):
            # print("current kwargs", kwargs)
            # print("reconstruct kwargs", self.reconstruct_kwargs)
            # raise ValueError("stop here")
            # print("returning cached")
            # check that the kwargs are the same
            if self.reconstruct_kwargs == kwargs:
                return self.cached_reconstruct
        if kwargs.get("cache", False):
            self.cache_reconstruct(**kwargs)
            return self.reconstruct(**kwargs)
        return self.reconstruct_(**kwargs)

    @torch.no_grad()
    def cache_reconstruct(self, offload: bool = False, **kwargs):
        # print("caching")
        self.register_buffer("cached_reconstruct", self.reconstruct_(**kwargs))
        self.reconstruct_kwargs = kwargs
        # if we offload, then we offload everything besides the cached reconstruct and bias
        if offload:
            original_device = self.cached_reconstruct.device
            self.to("cpu")
            self.cached_reconstruct = self.cached_reconstruct.to(original_device)
            if self.bias is not None:
                self.bias = nn.Parameter(
                    self.bias.data.detach().clone().to(original_device)
                )

    def delete_cache_reconstruct(self):
        del self.cached_reconstruct



    def clean(self):
        if hasattr(self, "original_weight"):
            del self.original_weight
        if hasattr(self, "hessian"):
            self.dump_hessian()
        if hasattr(self, "hessianDiag"):
            self.dump_hessianDiag()
        utils.recursive_apply(self, "clean")

    def get_n_original_parameters(self):
        return self.original_parameters

    def change_denormalization_method(
        self, new_method: Literal["otf", "reconstruct", "ignore"]
    ):
        self.denormalization_method = new_method

    def change_forward_method(self, new_method: Literal["reconstruct", "otf"]):
        self.forward_method = new_method

    def __str__(self):
        return self.name
