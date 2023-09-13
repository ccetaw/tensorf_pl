import torch
import torch.nn.functional as F

class AlphaGridMask(torch.nn.Module):
    """
    Grid mask indicating the occupancy of the space. 
    Used for filtering rays and points sampled outside the object.
    """
    def __init__(self, aabb, grid_size, threshold, device, logger):
        super().__init__()
        self.update_aabb(aabb, grid_size)
        self.device = device
        self.thres = threshold
        self.logger = logger

        self.alpha_volume = torch.ones(grid_size, device=device)

    def _compute_alpha_from_density_field(self, xyz_locs, density_field, step_size):
        # Filter out points outside original alpha grid mask to reduce computations.
        alphas = self.sample_alpha(xyz_locs)
        alpha_mask = alphas > 0

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device) 

        if alpha_mask.any():
            xyz_sampled = density_field.normalize_coord(xyz_locs[alpha_mask])
            sigma[alpha_mask] = density_field(xyz_sampled)

        alpha = 1 - torch.exp(-sigma*step_size) #.view(xyz_locs.shape[:-1])

        return alpha

    def sample_alpha(self, xyz_sampled):
        """
        Sample alpha(tranmittance) value using trilinear interpolation.
        ----
        Input:
        - xyz_sampled: Tensor [n_points, 3]. Spatial coordinates.
        """
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

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
        self.invaabb_size = 2.0/self.aabb_size
        self.units=self.aabbSize / (self.grid_size-1)
        self.aabb_diag = torch.sqrt(torch.sum(torch.square(self.aabb_size)))

    def update_alpha_volume(self, density_field, grid_size, step_size):
        """
        Update the alpha mask and get the new bounding box.
        Grid size changes dynamically during training.
        """
        grid_size = density_field.grid_size if grid_size is None else grid_size

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, grid_size[0]),
            torch.linspace(0, 1, grid_size[1]),
            torch.linspace(0, 1, grid_size[2]),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples # [grid_size[0], grid_size[1], grid_size[2], 3], gives the coordinates of grid points

        alpha = torch.zeros_like(dense_xyz[...,0]) # [grid_size[0], grid_size[1], grid_size[2]]
        for i in range(grid_size[0]): # Avoid oom by slicing
            alpha[i] = self._compute_alpha_from_density_field(dense_xyz[i].view(-1,3), density_field, step_size).view((grid_size[1], grid_size[2]))

        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]
        total_voxels = grid_size[0] * grid_size[1] * grid_size[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(grid_size[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0
        self.update_aabb(density_field.aabb, grid_size)
        self.alpha_volume = alpha.view(1,1,*alpha.shape[-3:]) # Additional dimension for grid_sample

        valid_xyz = dense_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        info = f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100)
        self.logger.debug_print(info)

        return new_aabb


    def normalize_coord(self, xyz_locs):
        """
        Normalize to [-1, 1]. Required by F.grid_sample().
        ----
        Input:
        - xyz_locs: Tensor [n_points, 3]. Spatial coordinates.
        """
        return (xyz_locs-self.aabb[0]) * self.invaabb_size - 1
