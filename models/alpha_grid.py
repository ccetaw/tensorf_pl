import torch
import torch.nn.functional as F
import numpy as np
from logger import logger

class AlphaGridMask():
    """
    Grid mask indicating the occupancy of the space. 
    Used for filtering rays and points sampled outside the object.
    Note that in this class state_dict() and load_state_dict() are implemented for consistent behaviors.
    """
    def __init__(self, aabb, grid_size, threshold, device, alpha_volume=None, initialized=False):
        """
        Input:
        - aabb: Tensor [2,3]. AABB.
        - grid_size: 
        - threshold:
        - device:
        - alpha_volume:
        - initialized:
        """
        super().__init__()
        self.device = device
        self.thres = threshold
        self.initialized = initialized 
        self.update_aabb(aabb, grid_size)

        if alpha_volume is None:
            self.alpha_volume = torch.ones(grid_size, device=device).view(1,1, *self.grid_size.tolist()[-3:]) 
        else:
            length = grid_size[0] * grid_size[1] * grid_size[2]
            self.alpha_volume = torch.from_numpy(np.unpackbits(alpha_volume)[:length].reshape(grid_size))
            self.alpha_volume = self.alpha_volume.view(1, 1, *alpha_volume.shape[-3:])

    def get_kwargs(self):
        """
        Return the model parameters. Will be saved to checkpoint.
        """
        alpha_volume = self.alpha_volume.bool().cpu().numpy()
        return {
            'aabb': self.aabb,
            'grid_size': self.grid_size.tolist(),
            'threshold': self.thres,
            'initialized': self.initialized,
            'alpha_volume': np.packbits(alpha_volume.reshape(-1))
        }

    def update_aabb(self, aabb, grid_size):
        """
        Set aabb and grid size, as well as other related values that will be used.
        ----
        Input:
        - aabb: Tensor [2, 3]. Top left and bottom right.
        - grid_size: List [3,]. Grid size.
        """
        self.aabb = torch.tensor(aabb, device=self.device) # Assure it's tensor
        self.grid_size= torch.LongTensor(grid_size).to(self.device)

        self.aabb_size = self.aabb[1] - self.aabb[0]
        self.invaabb_size = 2.0/self.aabb_size
        self.units=self.aabb_size / (self.grid_size-1)
        self.aabb_diag = torch.sqrt(torch.sum(torch.square(self.aabb_size)))

        info = f"{self.__class__.__name__} current grid size {self.grid_size.tolist()}, aabb {self.aabb.tolist()}"
        logger.info_print(info)

    @torch.no_grad()
    def _compute_alpha_from_density_field(self, xyz_locs, density_field, step_size):
        # Filter out points outside original alpha grid mask to reduce computations.
        alphas = self.sample_alpha(xyz_locs)
        alpha_mask = alphas > 0

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device) 

        if density_field.activation== 'relu':
            feature2density = F.relu
        elif density_field.activation == 'softplus':
            feature2density = lambda x: F.softplus(x + density_field.value_offset)
        else:
            print(f"Unrecognized activation {density_field.activation}")
            exit()

        if alpha_mask.any():
            xyz_sampled = density_field.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = density_field(xyz_sampled)
            validsigma = feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma

        alpha = 1 - torch.exp(-sigma*step_size) #.view(xyz_locs.shape[:-1])

        return alpha

    def sample_alpha(self, xyz_sampled):
        """
        Sample alpha(tranmittance) value using trilinear interpolation.
        ----
        Input:
        - xyz_sampled: Tensor [n_points, 3]. Spatial coordinates.
        """
        xyz_locs = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_locs.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

    @torch.no_grad()
    def update_alpha_volume(self, density_field, grid_size, step_size):
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
        self.initialized = True
        grid_size = density_field.grid_size if grid_size is None else grid_size # Use density field grid if None

        # Uniform grid sampling
        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, grid_size[0]),
            torch.linspace(0, 1, grid_size[1]),
            torch.linspace(0, 1, grid_size[2]),
        ), -1).to(self.device)
        dense_xyz = density_field.aabb[0] * (1-samples) + density_field.aabb[1] * samples # [grid_size[0], grid_size[1], grid_size[2], 3], gives the coordinates of grid points

        alpha = torch.zeros_like(dense_xyz[...,0]) # [grid_size[0], grid_size[1], grid_size[2]]
        for i in range(grid_size[0]): # Avoid oom by slicing
            alpha[i] = self._compute_alpha_from_density_field(dense_xyz[i].view(-1,3), density_field, step_size).view((grid_size[1], grid_size[2]))

        dense_xyz = dense_xyz.transpose(0,2).contiguous() # [grid_size[2], grid_size[1], grid_size[0], 3]
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None] # [1, 1, grid_size[2], grid_size[1], grid_size[0]]
        total_voxels = grid_size[0] * grid_size[1] * grid_size[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(grid_size[::-1])
        alpha[alpha>=self.thres] = 1
        alpha[alpha<self.thres] = 0
        self.update_aabb(density_field.aabb, grid_size)
        self.alpha_volume = alpha.view(1,1,*alpha.shape[-3:]) # Additional dimension for grid_sample

        logger.debug_print(alpha.shape)
        logger.debug_print(self.alpha_volume.shape)
        
        valid_xyz = dense_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        info = f"Occupancy grid bbox: {xyz_min.tolist(), xyz_max.tolist()} alpha rest %%%f"%(total/total_voxels*100)
        logger.info_print(info)

        return new_aabb

    def normalize_coord(self, xyz_locs):
        """
        Normalize to [-1, 1]. Required by F.grid_sample().
        ----
        Input:
        - xyz_locs: Tensor [n_points, 3]. Spatial coordinates.
        """
        return (xyz_locs-self.aabb[0]) * self.invaabb_size - 1
