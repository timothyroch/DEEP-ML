import torch

def add_bias(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    bb = torch.broadcast_to(b, x.size())
    return x+bb
