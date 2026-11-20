import torch
import torch.nn.functional as F

def conv2d_via_unfold(x, W, b, stride=1, padding=0):
    dimX = x.shape
    n, cin, hin, win = int(dimX[0]), int(dimX[1]), int(dimX[2]), int(dimX[3])
    dimW = W.shape
    cout, cin, kh, kw = int(dimW[0]), int(dimW[1]), int(dimW[2]), int(dimW[3])

    hout = int((hin+2*padding - kh)/stride)+1
    wout = int((win+2*padding - kw)/stride)+1

    unf = F.unfold(x, (kh, kw), padding=padding, stride=stride)
    wflat = W.reshape(cout, int(cin*kh*kw))
    unf = wflat @ unf
    unf = unf.reshape(n, cout, hout, wout)
    bb = b.reshape(1, cout, 1, 1)
    res = unf + bb
    return res
