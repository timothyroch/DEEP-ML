import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if p==0:
        return x
    if training:
        b = torch.ones(x.shape)
        b = (1-p)*b
        b = torch.bernoulli(b)
        x = b*x
        return (1/(1-p))*x
    else:
        return x
