import torch
import torch.nn as nn


def vector_diffs(lines):
    """
    Vector regularization.
    ----
    Input:
    - lines: List of vectors(1D Tensor). 
    """
    total = 0
    
    for idx in range(len(lines)):
        n_comp, n_size = lines[idx].shape[1:-1]
        
        dotp = torch.matmul(lines[idx].view(n_comp,n_size), lines[idx].view(n_comp,n_size).transpose(-1,-2)) # Covariance matrix
        non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1] # Extract non diagonal elements
        total = total + torch.mean(torch.abs(non_diagonal))
    return total



def L1_VM(planes, lines):
    """
    L1 regularization for VM decompositon. 
    ----
    Input:
    - planes: List of matrices(2D Tensor). VM decomposition planes.
    - lines: List of vectors(1D Tensor). VM decomposition lines.
    """
    total = 0
    for idx in range(len(planes)):
        total = total + torch.mean(torch.abs(planes[idx])) + torch.mean(torch.abs(lines[idx]))# + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.planes[idx]))
    return total


def _TVloss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])
    count_w = _tensor_size(x[:,:,:,1:])
    h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
    w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
    return 2*(h_tv/count_h+w_tv/count_w)/batch_size

def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]

def TVloss(planes):
    """
    Total variation loss. 
    ----
    Input:
    - planes: List of VM planes(Tensor[1, R, h, w])
    """
    total = 0
    for idx in range(len(planes)):
        total = total + _TVloss(planes[idx]) * 1e-2 #+ reg(self.density_line[idx]) * 1e-3
    return total






