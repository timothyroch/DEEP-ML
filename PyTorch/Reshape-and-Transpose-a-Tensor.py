import torch

def flatten_then_reshape(x: torch.Tensor, new_shape) -> torch.Tensor:
    flt = torch.flatten(x)
    return torch.reshape(flt, new_shape)

def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    ind = len(x.shape)
    return torch.transpose(x, ind-2, ind-1)
