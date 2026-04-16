import torch 
import torch.nn as nn 
from typing import List, Tuple



class BlockwiseDiagMatrix(nn.Module):
    def __init__(self, d, block_size:int = 64, initalize_as_identity:bool = False,
                    device: torch.device = None):
        
        super(BlockwiseDiagMatrix, self).__init__()
        self.d = d
        self.block_size = block_size
        assert d % block_size == 0, "Dimension must be divisible by block size"
        self.num_blocks = d // block_size
        if initalize_as_identity:
            # Initialize the diagonal blocks as identity matrices
            self.diag_blocks = nn.Parameter(torch.eye(block_size).repeat(self.num_blocks, 1, 1).to(device))
        else:
            # Initialize the diagonal blocks as random matrices
            self.diag_blocks = nn.Parameter(torch.randn(self.num_blocks, block_size, block_size).to(device))

    def forward(self, x, leading:bool = True):
        """
        Forward pass for the blockwise diagonal matrix.
        :param x: Input tensor of shape (..., d) if leading is False, or (..., d, m) if leading is True.
        :param leading: If True, treat the blockwise matrix as leading (left multiplication).
        :return: Output tensor after applying the blockwise diagonal matrix. should be of the same shape as x
        """
        if leading:
            #assumes its (block_matrix) @ x
            #equivalent to calling self.__matmul__(x)
            assert x.shape[-2] == self.d, "Input dimension must match the matrix dimension"
            # Perform blockwise multiplication
       
            result = torch.einsum('bik,...bkl->...bil', self.diag_blocks, x.reshape(x.shape[:-2] + (self.num_blocks, self.block_size, x.shape[-1])))
        else:
            #assumes its x @ (block_matrix)
            #equivalent to calling self.__rmatmul__(x)
            assert x.shape[-1] == self.d, "Input dimension must match the matrix dimension"
            # Perform blockwise multiplication
            result = torch.einsum('...bi,bij->...bj', x.reshape(-1, self.num_blocks, self.block_size), self.diag_blocks)
        return result.reshape_as(x)
        # else:
        #     return self.__rmatmul__(x)
    
    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2] == self.d, "Input dimension must match the matrix dimension"
        # Perform blockwise multiplication
        result = torch.einsum('bik,...bkl->...bil', self.diag_blocks, x.reshape(x.shape[:-2] + (self.num_blocks, self.block_size, x.shape[-1])))
        return result.reshape_as(x)
            
    def __rmatmul__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d, "Input dimension must match the matrix dimension"
        # Perform blockwise multiplication
        result = torch.einsum('...bi,bij->...bj', x.reshape(-1, self.num_blocks, self.block_size), self.diag_blocks)
        return result.reshape_as(x)
    
    def get_dense(self):
        return torch.block_diag(*[self.diag_blocks[i] for i in range(self.num_blocks)])
    
    def get_n_bits(self):
        return self.diag_blocks.numel() * 16  # Assuming 16 bits per element for float16
    
    
if __name__ == "__main__":
    # Example usage
    d = 128
    block_size = 32
    matrix = BlockwiseDiagMatrix(d=d, block_size=block_size, initalize_as_identity=False)
    
    # Create a random tensor with the appropriate shape
    tensor = torch.randn(10, d, 20)  # Batch size of 10, d=128, and 20 features
    matrix_dense = matrix.get_dense()  # Get the dense representation of the matrix
    # Perform matrix multiplication
    result = matrix @ tensor  # Calls matrix.__matmul__(tensor)
    result_dense = torch.matmul(matrix_dense, tensor)  # Dense multiplication for comparison
    assert torch.allclose(result, result_dense, atol=1e-5, rtol=1e-6
                          ), "Results do not match!"
    print(result.shape)  # Should be (10, d, 20)
    
    tensor = torch.randn(10, 20, d)  # Batch size of 10, 20 features, d=128
    result_r = tensor @ matrix  # Calls matrix.__rmatmul__(tensor)
    result_dense_r = torch.matmul(tensor, matrix_dense)  # Dense multiplication for comparison
    assert torch.allclose(result_r, result_dense_r, atol=1e-5), "Results do not match!"
    print(result_r.shape)  # Should be (10, d, 20)