import torch


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


