import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import logger

class TensorVMBase(nn.Module):
    """
    Class TensorBase. Parent class for Tensor3D and Tensor4D.
    """
    
    def __init__(self, 
                 aabb,           # Tensor [2, 3]. Axis aligned bounding box, containing bl and tr. 
                 grid_size,      # List of int. Grid size. 
                 n_comp,         # List of int. Number of decomposition components for each dimension.
                 value_offset,   # float. Offset to be added.
                 device
                 ) -> None:
        super().__init__()
        self.n_comp = n_comp
        self.value_offset = value_offset
        self.device = device

        self.aabb = torch.as_tensor(aabb, dtype=torch.float, device=self.device)
        self.grid_size = torch.as_tensor(grid_size, dtype=torch.int, device=self.device)
        self.update_aabb(self.aabb, self.grid_size)
        
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
            plane_coef.append(
                torch.nn.Parameter(
                    scale * torch.randn(
                        (1, self.n_comp[i], self.grid_size[mat_id_1].item(), self.grid_size[mat_id_0].item())
                    )
                )
            )  # 1 x R_x x size_y x size_z
            line_coef.append(
                torch.nn.Parameter(
                    scale * torch.randn(
                        (1, self.n_comp[i], self.grid_size[vec_id].item(), 1)
                    )
                )
            ) # 1 x R_x x size_x x 1

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
        self.aabb = torch.as_tensor(aabb).to(self.device)
        self.grid_size= torch.as_tensor(grid_size).to(self.device)

        aabb_size = self.aabb[1] - self.aabb[0]
        self.invaabb_size = 2.0/(aabb_size)
        self.units= (aabb_size)/ (self.grid_size-1)
        self.aabb_diag = torch.sqrt(torch.sum(torch.square(aabb_size)))

        info = f"{self.__class__.__name__} current grid size {self.grid_size.tolist()}, aabb {self.aabb.view(-1)}"
        logger.info_print(info)

    @torch.no_grad()
    def shrink(self, new_aabb, occupancy_grid):
        """
        Shrink the aabb to bound tighter. Keep units unchanged.
        ----
        Input:
        - new_aabb: Tensor [2, 3]. Top left and bottom right.
        """

        info = f"{self.__class__.__name__} shrinks to {new_aabb.view(-1)}"
        logger.info_print(info)

        # Select grid inside new_aabb
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.grid_size]).amin(0)


        for i in range(len(self.vec_mode)):
            mode0 = self.vec_mode[i]
            self.lines[i] = torch.nn.Parameter(
                self.lines[i].data[...,t_l[mode0]:b_r[mode0],:])
            mode0, mode1 = self.mat_mode[i]
            self.planes[i] = torch.nn.Parameter(
                self.planes[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]])
        
        # Clamp new_aabb to outer grid point
        if not torch.all(occupancy_grid.grid_size == self.grid_size):
            t_l_r, b_r_r = t_l / (self.grid_size-1), (b_r-1) / (self.grid_size-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            new_aabb = correct_aabb

        new_size = b_r - t_l

        self.update_aabb(new_aabb, (new_size[0], new_size[1], new_size[2]))

    @torch.no_grad()
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

        info = f"{self.__class__.__name__} upsample to {res_target}"
        logger.info_print(info)

        self.update_aabb(self.aabb, res_target)
    
 
class TensorVM3D(TensorVMBase):

    def __init__(self, aabb, grid_size, n_comp, value_offset, activation, device) -> None:
        super(TensorVM3D, self).__init__(aabb, grid_size, n_comp, value_offset, device)
        self.activation = activation  # Activation is not initialized here as there might be multiple decoders

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

        return feature


class TensorVM4D(TensorVMBase):
    
    def __init__(self, aabb, grid_size, n_comp, value_offset, activation, dim_4d, device) -> None:
        super(TensorVM4D, self).__init__(aabb, grid_size, n_comp, value_offset, device)

        self.activation = activation # Activation is not initialized here as there might be multiple decoders
        self.dim_4d = dim_4d
        self.basis_mat = torch.nn.Linear(sum(self.n_comp), self.dim_4d, bias=False).to(device)

    def forward(self, xyz_locs):
        """
        Trilinear interpolation. 
        ----
        Input:
        - xyz_locs: Tensor [n_points, 3]. Spatial coordinates

        Output:
        - feature: Tensor [n_points, dim_4d].
        """
        # plane + line basis
        coordinate_plane = torch.stack((xyz_locs[..., self.mat_mode[0]], xyz_locs[..., self.mat_mode[1]], xyz_locs[..., self.mat_mode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_locs[..., self.vec_mode[0]], xyz_locs[..., self.vec_mode[1]], xyz_locs[..., self.vec_mode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.planes)):
            plane_coef_point.append(F.grid_sample(self.planes[idx_plane], coordinate_plane[[idx_plane]],
                                                  align_corners=True).view(-1, *xyz_locs.shape[:1]))
            line_coef_point.append(F.grid_sample(self.lines[idx_plane], coordinate_line[[idx_plane]],
                                                 align_corners=True).view(-1, *xyz_locs.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)

        return self.basis_mat((plane_coef_point * line_coef_point).T) 
