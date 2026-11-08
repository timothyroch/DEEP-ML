import torch

def grad_of_quadratic(x_value: float) -> float:
    x_value= float(x_value)
    x = torch.tensor(x_value, requires_grad=True)
    f = x**2 + 3*x + 2
    f.backward()
    return float(x.grad)
