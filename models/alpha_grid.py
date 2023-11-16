import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from logger import logger

class AlphaGrid(nn.Module):
    """
    Grid mask indicating the occupancy of the space. 
    Used for filtering rays and points sampled outside the object.
    """
    def __init__(self, aabb, grid_size, threshold, device):
        """
        Input:
        - threshold: float. Mask threshold.
        """
        super().__init__()
        self.thres = threshold
        self.device = device
        
        if aabb is not None:
            self.aabb = torch.as_tensor(aabb, dtype=torch.float, device=self.device)
            self.grid_size = torch.as_tensor(grid_size, dtype=torch.int, device=self.device)
        else:
            self.aabb = torch.as_tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float, device=self.device)
            self.grid_size = torch.as_tensor([128, 128, 128], dtype=torch.int, device=self.device)
        self.update_aabb(self.aabb, self.grid_size.tolist())

        self.alpha_volume = None

    def update_aabb(self, aabb, grid_size):
        """
        Set aabb and grid size, as well as other related values that will be used.
        ----
        Input:
        - aabb: Tensor [2, 3]. Top left and bottom right.
        - grid_size: List [3,]. Grid size.
        """
        self.aabb = torch.as_tensor(aabb).to(self.aabb)
        self.grid_size= torch.as_tensor(grid_size).to(self.grid_size)

        aabb_size = self.aabb[1] - self.aabb[0]
        self.invaabb_size = 2.0/(aabb_size)

        info = f"{self.__class__.__name__} current grid size {self.grid_size}, aabb {self.aabb.view(-1)}"
        logger.info_print(info)

    def sample_alpha(self, xyz_locs):
        """
        Sample alpha(tranmittance) value using trilinear interpolation.
        ----
        Input:
        - xyz_sampled: Tensor [n_points, 3]. Spatial coordinates.
        """
        xyz_locs = self.normalize_coord(xyz_locs)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_locs.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_locs):
        """
        Normalize to [-1, 1]. Required by F.grid_sample().
        ----
        Input:
        - xyz_locs: Tensor [n_points, 3]. Spatial coordinates.
        """
        return (xyz_locs-self.aabb[0]) * self.invaabb_size - 1.0

    @torch.no_grad()
    def _compute_alpha_from_density_field(self, xyz_locs, DensityField, feature2density, step_size):
        # Filter out points outside original alpha grid mask to reduce computations.
        if self.alpha_volume is not None:
            alphas = self.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device) 

        if alpha_mask.any():
            xyz_sampled = DensityField.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = DensityField(xyz_sampled)
            validsigma = feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma

        alpha = 1 - torch.exp(-sigma*step_size).view(xyz_locs.shape[:-1])

        return alpha
    
    @torch.no_grad()
    def update_alpha_volume(self, DensityField, feature2density, grid_size, step_size):
        """
        Update the alpha mask and get the new bounding box.
        Grid size changes dynamically during training.
        ----
        Inputs:
        - density_field: Instance of class TensorVM3D. 
        - grid_size: List of ints. 
        - step_size: float.

        Outputs:
        - new_aabb: Tensor [2,3]. New AABB for density field. 
        """
        grid_size = DensityField.grid_size if grid_size is None else grid_size # Use density field grid if None

        # Uniform grid sampling
        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, grid_size[0]),
            torch.linspace(0, 1, grid_size[1]),
            torch.linspace(0, 1, grid_size[2]),
        ), -1).to(self.aabb)
        dense_xyz = DensityField.aabb[0] * (1-samples) + DensityField.aabb[1] * samples # [grid_size[0], grid_size[1], grid_size[2], 3], gives the coordinates of grid points

        alpha = torch.zeros_like(dense_xyz[...,0]) # [grid_size[0], grid_size[1], grid_size[2]]
        for i in range(grid_size[0]): # Avoid oom by slicing
            alpha[i] = self._compute_alpha_from_density_field(dense_xyz[i].view(-1,3), 
                                                              DensityField, 
                                                              feature2density, 
                                                              step_size).view((grid_size[1], grid_size[2]))
        
        dense_xyz = dense_xyz.transpose(0,2).contiguous() # [grid_size[2], grid_size[1], grid_size[0], 3]
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None] # [1, 1, grid_size[2], grid_size[1], grid_size[0]]
        total_voxels = grid_size[0] * grid_size[1] * grid_size[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(grid_size[::-1]) # [grid[2], grid[1], grid[0]]
        alpha[alpha>=self.thres] = 1
        alpha[alpha<self.thres] = 0
        self.update_aabb(DensityField.aabb, grid_size)
        self.alpha_volume = alpha.view(1, 1, *alpha.shape[-3:]) # Additional dimension for grid_sample
        
        valid_xyz = dense_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        info = f"New bbox: {new_aabb.view(-1)} alpha rest %%%f"%(total/total_voxels*100)
        logger.info_print(info)

        return new_aabb
