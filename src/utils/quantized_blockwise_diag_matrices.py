import torch 
import torch.nn as nn 
from typing import List, Tuple, Literal
from src.utils.blockwise_diag_matricies import BlockwiseDiagMatrix

class QuantizedBlockwiseDiagMatrix(BlockwiseDiagMatrix):
    def __init__(self, d, block_size:int = 64, initalize_as_identity:bool = False,
                    device: torch.device = None, quant_precision: int = 4):
        super().__init__(d, block_size, initalize_as_identity, device)
        self.quant_precision = quant_precision
        self.quant_range = (-2 ** (quant_precision - 1), 2 ** (quant_precision - 1) - 1)

        #group_size = self.block_size
        self.scales = nn.Parameter(self.diag_blocks.abs().amax(dim=-1, keepdim=True) / self.quant_range[1])
        
        #keep self.diag_blocks as float master copy
        #mimic QuantizedSparseLinear

    def get_dequantized(self, method: Literal["super", "straight_through", "standard"] = "straight_through"):
        if method == "super":
            return self.diag_blocks
        elif method == "standard":
            with torch.no_grad():
                return torch.round(self.diag_blocks / self.scales).clamp(*self.quant_range) * self.scales
        else:
            diag_blocks_scaled = self.diag_blocks / self.scales
            diag_blocks_ste = ((torch.round(diag_blocks_scaled) - diag_blocks_scaled).detach() + diag_blocks_scaled).clamp(*self.quant_range)
            return diag_blocks_ste * self.scales

    def get_n_bits(self):
        return self.diag_blocks.numel() * self.quant_precision + self.scales.numel() * 16
           