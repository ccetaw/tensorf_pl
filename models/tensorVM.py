import torch
import torch.nn as nn
import torch.nn.functional as F

class TensorVMBase(nn.Module):
    """
    Class TensorBase. Parent class for Tensor3D and Tensor4D
    """
    
    def __init__(self, 
                 aabb,           # Tensor [2, 3]. Axis aligned bounding box, containing bl and tr. 
                 grid_size,      # List of int. Grid size. 
                 n_comp,         # List of int. Number of decomposition components for each dimension.
                 value_offset,   # float or vector. Offset to be added.
                 device,         # str. 'cuda' or 'cpu'.
                 logger     # Logger. Customized Logger class. 
                 ) -> None:
        super().__init__()
        self.update_aabb(aabb, grid_size)
        self.n_comp = n_comp
        self.value_offset = value_offset
        self.device = device
        self.logger = logger

        self.mat_mode = [[0,1], [0,2], [1,2]]
        self.vec_mode =  [2, 1, 0]

        self.planes, self.lines = self.init_VM_decomp(scale=0.1)

    def init_VM_decomp(self, scale):
        """
        Grid values initialization using random numbers between scale*[0,1].
        """
        plane_coef, line_coef = [], []
        for i in range(len(self.vec_mode)):
            vec_id = self.vec_mode[i]
            mat_id_0, mat_id_1 = self.mat_mode[i]

            # Additional dimension to match the input of F.grid_sample
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, self.n_comp[i], self.grid_size[mat_id_1], self.grid_size[mat_id_0]))))  # 1 x R_x x size_y x size_z
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, self.n_comp[i], self.grid_size[vec_id], 1)))) # 1 x R_x x size_x x 1

        return torch.nn.ParameterList(plane_coef).to(self.device), torch.nn.ParameterList(line_coef).to(self.device) # Parameters moved to device

    def normalize_coord(self, xyz_locs):
        """
        Normalize to [-1, 1].
        ----
        Input:
        - xyz_locs: Tensor [n_points, 3]. Spatial coordinates.
        """
        return (xyz_locs-self.aabb[0]) * self.invaabb_size - 1

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

    def shrink(self, new_aabb):
        """
        Shrink the aabb to bound tighter. Keep units unchanged.
        ----
        Input:
        - new_aabb: Tensor [2, 3]. Top left and bottom right.
        """

        # Select grid inside new_aabb
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.grid_size]).amin(0)

        # ---- DEBUG ---- #
        self.logger.debug_print(xyz_min)
        self.logger.debug_print(xyz_max)
        self.logger.debug_print(t_l)
        self.logger.debug_print(b_r)
        # ---- DEBUG ---- #

        for i in range(len(self.vec_mode)):
            mode0 = self.vec_mode[i]
            self.lines[i] = torch.nn.Parameter(
                self.lines[i].data[...,t_l[mode0]:b_r[mode0],:])
            mode0, mode1 = self.mat_mode[i]
            self.planes[i] = torch.nn.Parameter(
                self.planes[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]])
        
        # Clamp new_aabb to outer grid point
        t_l_r, b_r_r = t_l / (self.grid_size-1), (b_r-1) / (self.grid_size-1)
        correct_aabb = torch.zeros_like(new_aabb)
        correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
        correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]

        # ---- DEBUG ---- #
        self.logger.debug_print(new_aabb)
        self.logger.debug_print(correct_aabb)
        # ---- DEBUG ---- #

        new_aabb = correct_aabb
        new_size = b_r - t_l

        self.update_aabb(new_aabb, new_size)

    def upsample(self, res_target):
        """
        Upsample the grid to the target resolution. Up sample planes and lines separately. 
        ----
        Input:
        - res_target: List [3,]. Target resolution.
        """
        for i in range(len(self.vec_mode)):
            vec_id = self.vec_mode[i]
            mat_id_0, mat_id_1 = self.mat_mode[i]
            self.planes[i] = torch.nn.Parameter(
                F.interpolate(self.planes[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            self.lines[i] = torch.nn.Parameter(
                F.interpolate(self.lines[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        # ---- DEBUG ---- #
        self.logger.debug_print(self.planes[0].shape)
        self.logger.debug_print(self.lines[0].shape)
        # ---- DEBUG ---- #

        self.update_aabb(self.aabb, res_target)

    
 
class TensorVM3D(TensorVMBase):

    def __init__(self, aabb, grid_size, n_comp, value_offset, device, activation, logger) -> None:
        super().__init__(aabb, grid_size, n_comp, value_offset, device, logger)
        self.activation = activation

    def get_kwargs(self):
        """
        Return the model hyperparameters. Will be saved to checkpoint.
        """
        return {
            'aabb': self.aabb,
            'grid_size': self.grid_size.tolist(),
            'n_comp': self.density_n_comp,
            'dim': self.app_dim,
            'value_offset': self.value_offset,
            'activation': self.activation
        }

    def forward(self, xyz_locs):
        """
        Trilinear interpolation. 
        ----
        Input:
        - xyz_locs: Tensor [n_points, 3]. Spatial coordinates

        Output:
        - feature: Tensor [n_points, ]
        """
        # plane + line basis
        coordinate_plane = torch.stack((xyz_locs[..., self.mat_mode[0]], xyz_locs[..., self.mat_mode[1]], xyz_locs[..., self.mat_mode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_locs[..., self.vec_mode[0]], xyz_locs[..., self.vec_mode[1]], xyz_locs[..., self.vec_mode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        # ---- DEBUG ---- #
        self.logger.debug_print(coordinate_plane.shape)
        self.logger.debug_print(coordinate_line.shape)
        # ---- DEBUG ---- #

        feature = torch.zeros((xyz_locs.shape[0],), device=xyz_locs.device)
        for idx_plane in range(len(self.planes)):
            """
            torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=none)
            `grid` specifies the sampling pixel locations normalized by the input spatial dimensions. therefore, it should have most values in the range of [-1, 1].
            """
            plane_coef_point = F.grid_sample(self.planes[idx_plane], coordinate_plane[[idx_plane]], # double brackets to keep dimensions
                                             align_corners=True).view(-1, *xyz_locs.shape[:1])
            # [1, n_comp, n_points = n_rays * n_samples, 1] = grid_sample([1, n_comp, grid_x, grid_y], [1, n_points, 1, 2])
            # viewed as [n_comp, n_points]
            line_coef_point = F.grid_sample(self.lines[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_locs.shape[:1])
            feature = feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        # ---- DEBUG ---- #
            self.logger.debug_print(plane_coef_point.shape)
            self.logger.debug_print(line_coef_point.shape)
            self.logger.debug_print(feature.shape)
        # ---- DEBUG ---- #

        if self.activation is not None:
            feature = self.activation(feature, self.value_offset) # offset will be added for softplus activation but not for relu

        return feature


class TensorVM4D(TensorVMBase):
    
    def __init__(self, aabb, grid_size, n_comp, value_offset, device, activation, logger, dim_4d=1) -> None:
        super().__init__(aabb, grid_size, n_comp, value_offset, device, logger)
        self.activation = activation

        self.dim_4d = dim_4d
        self.basis_mat = torch.nn.Linear(sum(self.n_comp), self.dim_4d, bias=False).to(device)

    def get_kwargs(self):
        """
        Return the model hyperparameters. Will be saved to checkpoint.
        """
        return {
            'aabb': self.aabb,
            'grid_size': self.grid_size.tolist(),
            'n_comp': self.density_n_comp,
            'dim': self.app_dim,
            'value_offset': self.value_offset,
            'dim_4d': self.dim_4d,
            'activation': self.activation
        }

    def forward(self, xyz_locs):
        """
        Trilinear interpolation. 
        ----
        Input:
        - xyz_locs: Tensor [n_points, 3]. Spatial coordinates
        """
        # plane + line basis
        coordinate_plane = torch.stack((xyz_locs[..., self.mat_mode[0]], xyz_locs[..., self.mat_mode[1]], xyz_locs[..., self.mat_mode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_locs[..., self.vec_mode[0]], xyz_locs[..., self.vec_mode[1]], xyz_locs[..., self.vec_mode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        # ---- DEBUG ---- #
        self.logger.debug_print(coordinate_plane.shape)
        self.logger.debug_print(coordinate_line.shape)
        # ---- DEBUG ---- #

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.planes)):
            plane_coef_point.append(F.grid_sample(self.planes[idx_plane], coordinate_plane[[idx_plane]],
                                                  align_corners=True).view(-1, *xyz_locs.shape[:1]))
            line_coef_point.append(F.grid_sample(self.lines[idx_plane], coordinate_line[[idx_plane]],
                                                 align_corners=True).view(-1, *xyz_locs.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)

        feature = torch.zeros(xyz_locs.shape[0], self.dim_4d)
        feature = self.basis_mat((plane_coef_point * line_coef_point).T) 

        # ---- DEBUG ---- #
        self.logger.debug_print(plane_coef_point)
        self.logger.debug_print(line_coef_point)
        self.logger.debug_print(feature)
        # ---- DEBUG ---- #

        if self.activation is not None:
            feature = self.activation(feature, self.value_offset)

        return feature
