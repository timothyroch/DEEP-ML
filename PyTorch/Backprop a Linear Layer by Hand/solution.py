import torch

def linear_backward(grad_output, x, W):
    y = x@W.T

    grad_input = grad_output @ W
    grad_W = grad_output.T @ x
    grad_b = grad_output
    gb = torch.sum(grad_b, dim=0)

    return (grad_input, grad_W, gb)
