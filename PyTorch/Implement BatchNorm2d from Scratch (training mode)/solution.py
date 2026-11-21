import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    n, c, h, w = x.shape

    mn = torch.mean(x, keepdim=True, dim=(0, 2, 3))
    vr = torch.var(x, keepdim=True, dim=(0, 2, 3), correction=0)
    std = torch.sqrt(vr+eps)

    gamma = gamma.view(1, c, 1, 1)
    beta = beta.view(1, c, 1, 1)

    Xnchw = (x-mn)/std

    Ynchw = gamma*Xnchw + beta

    return Ynchw
