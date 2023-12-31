import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from logger import logger
from .tensorVM import TensorVM3D, TensorVM4D
from .alpha_grid import AlphaGrid
from .sh import eval_sh_bases


def positional_encoding(positions, freqs):
    """
    Original NeRF sinusoidal positional encoding.
    ----
    Inputs:
    - positions: Tensor [n_points, 3]. Spatial coordinates.
    - freqs: int. 

    Outputs:
    - pts: Tensor [n_points, 2*freqs]. Encodings. 
    """
    
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


def raw2alpha(sigma, dist):
    """
    Turn density into transmittance.
    ----
    Inputs:
    - sigma: Tensor [n_rays, n_points]. Density values.
    - dist: Tensor [n_rays, n_points]. Distance values.

    Outputs:
    - alpha: Tensor [n_rays, n_points]. Transmittance of invervals.
    - weights: Tensor [n_rays, n_points]. Weights.
    - background_weight: Tensor [n_rays, n_points]. Transmittance to background.
    """
    # sigma, dist  [N_rays, n_sample]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, n_sample]
    return alpha, weights, T[:,-1:]


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb



"""
Three types of MLP that turn appearance feature to radiance are implemented.
Directions are by default positionally encoded.
"""

class MLP(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, featureC=128):
        super(MLP, self).__init__()

        self.in_mlpC = (3+2*viewpe*3) + inChanel
        self.viewpe = viewpe
        
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLP_Fea(torch.nn.Module):
    """
    Features are positional encodings.
    """
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):
        super(MLP_Fea, self).__init__()

        self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLP_PE(torch.nn.Module):
    """
    Spatial coordinates positional encodings.
    """
    def __init__(self,inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLP_PE, self).__init__()

        self.in_mlpC = (3+2*viewpe*3)+ (3+2*pospe*3)  + inChanel #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb



class NeRFRenderer(nn.Module):
    """
    NeRF-style volumetric rendering. 
    """

    def __init__(self,
                 near_far,
                 step_ratio,
                 distance_scale,
                 n_samples,
                 white_bg,
                 ndc_ray,
                 ray_march_weight_thres,
                 density_field_config,
                 radiance_field_config,
                 occupancy_grid_config,
                 device) -> None:
        super().__init__()
        self.near_far = near_far
        self.step_ratio = step_ratio
        self.distance_scale = distance_scale
        self.ray_march_weight_thres = ray_march_weight_thres
        self.n_samples = n_samples
        self.white_bg = white_bg,
        self.ndc_ray = ndc_ray
        self.device = device

        self.DensityField = TensorVM3D(**density_field_config, device=device)
        self.RadianceField = TensorVM4D(**radiance_field_config, device=device)
        self.OccupancyGrid = AlphaGrid(**occupancy_grid_config, device=device)

        self.update_stepsize()
        self.init_activations()

    def init_activations(self):
        # density activation
        density_activation = self.DensityField.activation  
        if density_activation == 'relu':
            self.feature2density = F.relu
        elif density_activation == 'softplus':
            self.feature2density = lambda x: F.softplus(x + self.DensityField.value_offset)
        else:
            print(f"Unrecognized activation {density_activation}")
            exit()

        # radiance activation
        rgb_activation = self.RadianceField.activation
        if rgb_activation['MLP'] == 'MLP':
            self.feature2rgb = MLP(self.RadianceField.dim_4d, rgb_activation['view_pe'], rgb_activation['featureC']).to(self.device)
        elif rgb_activation['MLP'] == 'MLP_PE':
            self.feature2rgb = MLP_PE(self.RadianceField.dim_4d, rgb_activation['view_pe'], rgb_activation['pos_pe'], rgb_activation['featureC']).to(self.device)
        elif rgb_activation['MLP'] == 'MLP_Fea':
            self.feature2rgb = MLP_Fea(self.RadianceField.dim_4d, rgb_activation['view_pe'], rgb_activation['fea_pe'], rgb_activation['featureC']).to(self.device)
        elif rgb_activation['MLP'] == 'SH':
            self.feature2rgb = SHRender
        else:
            print(f"Unrecognized activation {rgb_activation['MLP']}")
            exit()

    def update_stepsize(self):
        self.step_size=torch.mean(self.DensityField.units)*self.step_ratio
        self.n_samples=int((self.DensityField.aabb_diag / self.step_size).item()) + 1

        info = f"Renderer: step size {self.step_size}, n_samples {self.n_samples}"
        logger.info_print(info)

    def get_kwargs(self):
        """
        Return the model hyperparameters. Will be saved to checkpoint.
        """
        return {
            'near_far': self.near_far,
            'step_ratio': self.step_ratio,
            'distance_scale': self.distance_scale,
            'n_samples': self.n_samples,
            'ray_march_weight_thres': self.ray_march_weight_thres
        }

    def get_opt_params(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        """
        Set different learning rate for different parameters.
        """
        grad_vars = [{'params': self.DensityField.lines, 'lr': lr_init_spatialxyz}, {'params': self.DensityField.planes, 'lr': lr_init_spatialxyz},
                     {'params': self.RadianceField.lines, 'lr': lr_init_spatialxyz}, {'params': self.RadianceField.planes, 'lr': lr_init_spatialxyz},
                         {'params': self.RadianceField.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.feature2rgb, torch.nn.Module):
            grad_vars += [{'params':self.feature2rgb.parameters(), 'lr':lr_init_network}]
        return grad_vars

    @torch.no_grad()
    def filter_rays(self, all_rays, all_rgbs, chunk=10240*5, bbox_only=False):
        """
        Filter out rays that do not hit the bounding box
        """
        info = "Filtering rays...."
        logger.info_print(info)
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.DensityField.aabb[1] - rays_o) / vec
                rate_b = (self.DensityField.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
                mask_inbbox = t_max > t_min
            else:
                xyz_sampled, _, _ = self.stratified_sample_points(rays_o, rays_d, is_train=False)
                mask_inbbox= (self.OccupancyGrid.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered]

    def stratified_sample_points(self, rays_o, rays_d, is_train=True):
        """
        Sample points on rays. Use stratified sampling during training and uniform sampling during evaluation.
        ----
        Inputs:
        - rays_o: Tensor [batch_size, 3]. Ray origins.
        - rays_d: Tensor [batch_size, 3]. Ray directions.
        - is_train: bool. 

        Outputs:
        - rays_pts: Tensor [batch_size, n_samples, 3]. Spatial coordinates.
        - interpx: Tensor []. 
        - mask_inbox: bool Tensor [batch_size, n_samples]. Indicating in/outside bounding box. 
        """
        n_samples = self.n_samples

        step_size = self.step_size # Same step size for every ray
        near, far = self.near_far

        # ray-AABB intersection
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.DensityField.aabb[1] - rays_o) / vec
        rate_b = (self.DensityField.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        # Sample points uniformly
        rng = torch.arange(n_samples)[None].float() # Random number generator
        if is_train:
            # Jitter
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]])
        step = step_size * rng.to(rays_o.device)
        interpx = (t_min[...,None] + step)
        
        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None] # [batch_size, n_samples, 3]
        mask_outbbox = ((self.DensityField.aabb[0]>rays_pts) | (rays_pts>self.DensityField.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    def startified_sample_points_ndc(self, rays_o, rays_d, is_train=True, n_samples=-1):
        """
        Sample points on rays in NDC space. Use stratified sampling during training and uniform sampling during evaluation.
        ----
        Inputs:
        - rays_o: Tensor [batch_size, 3]. Ray origins.
        - rays_d: Tensor [batch_size, 3]. Ray directions.
        - is_train: bool. 
        - n_samples: int. Number of points to be sampled.

        Outputs:
        - rays_pts: Tensor [batch_size, n_samples, 3]. Spatial coordinates.
        - interpx: Tensor []. 
        - mask_inbox: bool Tensor [batch_size, n_samples]. Indicating in/outside bounding box. 
        """
        n_samples = self.n_samples
        near, far = self.near_far
        interpx = torch.linspace(near, far, n_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / n_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.DensityField.aabb[0] > rays_pts) | (rays_pts > self.DensityField.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def _forward(self, rays_chunk):
        """
        NeRF-style volumetric rendering. Render rgb colors as well as other values given rays.
        ----
        Inputs:
        - rays_chunk: Tensor [chunk_size, 6]. Ray origins and directions.

        Outputs:
        - rgb_map: Tensor [chunk_size, 3]. RGB colors.
        - depth_map: Tensor [chunk_size, 1]. Depth.
        """

        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if self.ndc_ray:
            # Sample points on rays and fitler out points not inside the bounding box
            xyz_sampled, z_vals, ray_valid = self.startified_sample_points_ndc(rays_chunk[:, :3], viewdirs, is_train=self.training)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.stratified_sample_points(rays_chunk[:, :3], viewdirs, is_train=self.training)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        
        # Again filter points using OccupancyGrid
        if self.OccupancyGrid.alpha_volume is not None: 
            # Filter out points with invalid alpha values
            alphas = self.OccupancyGrid.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask) 
            ray_valid = ~ray_invalid

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        # Calcultate density at sampled points
        if ray_valid.any():
            xyz_sampled = self.DensityField.normalize_coord(xyz_sampled) # Normalized for both density and radiance field. 
            sigma_feature = self.DensityField(xyz_sampled[ray_valid])
            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.ray_march_weight_thres

        # Caclultate radiance at sampled points. 
        if app_mask.any():
            app_features = self.RadianceField(xyz_sampled[app_mask])
            valid_rgbs = self.feature2rgb(xyz_sampled[app_mask], viewdirs[app_mask], app_features)
            rgb[app_mask] = valid_rgbs

        # Accumuluate weighted radiance. 
        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if self.white_bg or (self.training and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])
        
        rgb_map = rgb_map.clamp(0,1)

        with torch.no_grad(): # No depth supervision.
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        out = {
            'rgb_map': rgb_map,
            'depth_map': depth_map,
            'acc_map': acc_map
        }

        return out

    def chunked_render(self, rays, chunk=4096):
        rgb_maps, depth_maps, acc_maps = [], [], []
        N_rays_all = rays.shape[0]
        for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
            rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk]
        
            out = self._forward(rays_chunk)

            rgb_maps.append(out["rgb_map"])
            depth_maps.append(out["depth_map"])
            acc_maps.append(out["acc_map"])
        
        return {
            "rgb_map": torch.cat(rgb_maps, dim=0),
            'depth_map': torch.cat(depth_maps, dim=0),
            'acc_map': torch.cat(acc_maps, dim=0)
        }

    def forward(self, rays):
        if self.training:
            out = self._forward(rays)
        else:
            out = self.chunked_render(rays)

        return out

class PBRenderer(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        pass

