import torch

def linear_forward(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    wt = torch.transpose(W, 0, 1)
    xWt = torch.matmul(x, wt)
    return xWt+b
