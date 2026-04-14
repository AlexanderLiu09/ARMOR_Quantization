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
from src.ARMOR_compress import initalize_optimizer, TrainingConfig
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


@dataclass
class SelectionConfig:
    random: bool = False #random or gradient based
    masked: bool = False #whether to only select from the masked elements
    norm: Literal["L1, L2", "Linf"] = "L1" #the norm to use for the gradient based selection
    greedy: bool = True #whether to use greedy or random selection for the gradient based selection
    
    @classmethod
    def from_name(cls, name: str):
        #if the name is just random
        if name == "random":
            return cls(random=True, masked=False)
        prefix,name = name.split("_",1)
        assert prefix=="gradient", f"Unknown selection method {name}, expected 'random' or 'gradient_*'."
        
        terms = name.split("_")
        defaults = {"masked": False, "norm": "L1", "greedy": False}
        for term in terms:
            if term == "masked":
                defaults["masked"] = True
            if term == "all":
                defaults["masked"] = False
            elif term in ["L1", "L2", "Linf"]:
                defaults["norm"] = term
            elif term in ["greedy", "random"]:
                defaults["greedy"] = term == "greedy"
            else:
                raise ValueError(f"Unknown selection method {name}, expected 'random' or 'gradient_*_masked_*_{'L1'|'L2'|'Linf'}'.")
        
        return cls(random=False, **defaults)
    


@torch.no_grad()         
def sparse_core_step(trainable_sparse: BlockCompressLearnable,
                            init_loss: float,
                            n_times: int = 1,
                            select: Literal["random", "gradient_greedy", "gradient_random"] = "random",
                            quant_config: Optional[QuantConfig] = None)-> None:
    
    
    """Performs n_times of discrete optimization on the sparse core of the compression module.

    Args:
        trainable_sparse (BlockCompressLearnable): The trainable overall module representing the ARMOR decomposition.
        n_times (int, optional): The number of times to perform discrete optimization. Defaults to 1.
        select (Literal["random", "gradient_greedy", "gradient_random"], optional): The method to use for selecting the groups to update. Defaults to "random".
    """
    raise NotImplementedError("sparse core step is not implemented yet, we need to implement the corresponding version for quantized sparse linear")
        
                
        
        
        
        
        
        
        
        


        



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
            prev_iter_loss = trainable_sparse.recon_loss(reduction="mean").item()
            if self.verbose:
                print("Pre-quantization loss (using original sparse reconstruction):", pre_quant_loss)
                print(f"Initial loss: {prev_iter_loss}")
            if self.use_wandb:
                self.wandb_queue.put({"self.metric_name": pre_quant_loss,
                            self.step_metric: 0})
                self.wandb_queue.put({self.metric_name: prev_iter_loss,
                           self.step_metric: 1})
            # best_state_dict = copy.deepcopy(trainable_sparse.state_dict())
            
        for i in tqdm.tqdm(range(training_config.n_iters), disable = not self.verbose):
            
            #optimizer step
            for j in tqdm.tqdm(range(training_config.n_continous_updates_per_iter),  disable = (not self.verbose or training_config.n_continous_updates_per_iter<10)):
                
                #reset the optimizers
                optimizer.zero_grad()   
                
                recon_loss = trainable_sparse.recon_loss(reduction="mean") 
                loss = recon_loss
             
                loss.backward()
                #step the optimizers
                optimizer.step()
                    

            if training_config.n_sparse_core_updates_per_iter != 0:
                with torch.no_grad():
                    sparse_core_step(
                        trainable_sparse,
                        n_times=training_config.n_sparse_core_updates_per_iter,
                        select=training_config.sparse_core_step_select,
                    )
                    # raise ValueError("stop here, we are done with training")
             #loss stuff
            with torch.no_grad():
                current_loss = trainable_sparse.recon_loss(reduction="mean").item()
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
        self.naive_compression_module.uncompress_sparse_values()
        self.naive_compression_module.load_state_dict(state_dict["naive_compression_module"])
        self.naive_compression_module.compress_sparse_values()
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

        if training_config.quant_enabled:
            print("Loading Quantized Weights")
            naive_cfg = copy.deepcopy(naive_compression_config)
            with open_dict(naive_cfg):
                naive_cfg.init_config._target_ = "src.sparse_compress.QuantizedSparseLinear"
                naive_cfg.compression_config.quant_n_bits = training_config.quant_n_bits
                naive_cfg.compression_config.quant_n_grid = training_config.quant_n_grid
            
            self.naive_compression_module = utils.blank_init(
                naive_cfg,
                n_in = self.original_weight.shape[1],
                n_out = self.original_weight.shape[0],
                dtype=self.original_weight.dtype,
                device=self.original_weight.device,)

        else:
            print("Loading FP16 Weights")
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
        proj_name = "layer_1/mlp.down_proj"
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