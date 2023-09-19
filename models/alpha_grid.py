import torch
import torch.nn.functional as F
import numpy as np
from logger import logger

class AlphaGrid():
    """
    Grid mask indicating the occupancy of the space. 
    Used for filtering rays and points sampled outside the object.
    """
    def __init__(self, aabb, grid_size, threshold, device, alpha_volume=None):
        """
        Input:
        - aabb: Tensor [2,3]. AABB.
        - grid_size: List of ints [3,]. 
        - threshold: float. Mask threshold.
        - device: str. 
        - alpha_volume: 
        - initialized:
        """
        super().__init__()
        self.device = device
        self.thres = threshold
        if aabb is not None and grid_size is not None:
            # Initially alpha grid does not exist as we don't have a density field.
            self.update_aabb(aabb, grid_size)

        if alpha_volume is not None: 
            length = grid_size[0] * grid_size[1] * grid_size[2]
            self.alpha_volume = torch.from_numpy(np.unpackbits(alpha_volume)[:length].reshape(grid_size))
            self.alpha_volume = self.alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        else:
            self.alpha_volume = None


    def get_kwargs(self):
        """
        Return the model parameters. Will be saved to checkpoint.
        """
        alpha_volume = self.alpha_volume.bool().cpu().numpy()
        return {
            'aabb': self.aabb,
            'grid_size': self.grid_size.tolist(),
            'threshold': self.thres,
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
        self.aabb = aabb
        self.grid_size= torch.LongTensor(grid_size).to(self.device)

        self.aabb_size = self.aabb[1] - self.aabb[0]
        self.invaabb_size = 1.0/self.aabb_size * 2

        info = f"{self.__class__.__name__} current grid size {self.grid_size.tolist()}, aabb {self.aabb.view(-1)}"
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
