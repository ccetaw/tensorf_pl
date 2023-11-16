import pytorch_lightning as pl 
import torch.nn.functional as F
import torch
import numpy as np
from utils import cal_n_samples, N_to_reso, visualize_depth_numpy
from loss import vector_diffs, L1_VM, TVloss
from metrics import rgb_ssim, rgb_lpips, mse2psnr
from dataLoader.ray_utils import get_rays
from models.renderer import NeRFRenderer
from logger import logger

class TensoRF(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.batch_size = config["batch_size"]
        self.rank = config["device"]
        self.reso_cur = config["reso_cur"]
        self.reso_mask = self.reso_cur
        self.n_samples = config["n_samples"]
        if config["lr_decay_iters"] > 0:
            self.lr_factor = config["lr_decay_target_ratio"]**(1/config["lr_decay_iters"])
        else:
            config["lr_decay_iters"] = config["n_iters"]
            self.lr_factor = config["lr_decay_target_ratio"]**(1/config["n_iters"])

        self.config = config
        self.weights_dict = {
            'Ortho_reg_weight': config["Ortho_weight"],
            'L1_reg_weight': config["L1_weight_inital"],
            'TV_weight_density': config["TV_weight_density"],
            'TV_weight_app': config["TV_weight_app"]
        }

        self.upsamp_list = config["upsamp_list"]
        self.update_AlphaMask_list = config["update_AlphaMask_list"]
        self.N_voxel_list = (
            torch.round(
                torch.exp(
                    torch.linspace(
                        np.log(config["N_voxel_init"]), 
                        np.log(config["N_voxel_final"]), 
                        len(self.upsamp_list)+1
                    )
                )
            ).long()
        ).tolist()[1:] #linear in logrithmic space

        density_field_config = {
            'aabb': config["aabb"],
            'grid_size': config["reso_cur"],
            'n_comp': config["n_lamb_sigma"],
            'value_offset': config["density_shift"],
            'activation': config["fea2denseAct"]
        }

        radiance_field_config = {
            'aabb': config["aabb"],
            'grid_size': config["reso_cur"],
            'n_comp': config["n_lamb_sh"],
            'value_offset': None,
            'activation': {
                'MLP': config["shadingMode"],
                'pos_pe': config["pos_pe"],
                'view_pe': config["view_pe"],
                'fea_pe': config["fea_pe"],
                'featureC': config["featureC"]
            },
            'dim_4d': config["data_dim_color"],
        }

        occupancy_grid_config = {
            'aabb': config["OccupancyGrid_aabb"],
            'grid_size': config["OccupancyGrid_grid_size"],
            'threshold': config["alpha_mask_thre"],
        }

        renderer_config = {
            'near_far': config["near_far"],
            'white_bg': config["white_bg"],
            'ndc_ray': config['ndc_ray'],
            'step_ratio': config["step_ratio"],
            'distance_scale': config["distance_scale"],
            'n_samples': config["n_samples"],
            'ray_march_weight_thres': config["rm_weight_mask_thre"],
            'density_field_config': density_field_config,
            'radiance_field_config': radiance_field_config,
            'occupancy_grid_config': occupancy_grid_config,
            'device': self.rank
        }

        self.model = eval(config["model_name"])(**renderer_config)
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def preprocess_data(self, batch, stage):
        assert stage in ['train', 'val', 'test', 'predict']
        if 'index' in batch: # validation / test / predict
            index = batch['index'].cpu()
        if stage == 'train':
            if (
                not hasattr(self, "rays")
                and not hasattr(self, "rgbs")
            ):
                self.rays = self.dataset.all_rays
                self.rgbs = self.dataset.all_rgbs

            if self.global_step == self.update_AlphaMask_list[1]:
                self.rays, self.rgbs = self.model.filter_rays(
                    self.rays,
                    self.rgbs,
                    bbox_only=self.global_step == 0,
                )

            index = torch.randint(
                0,
                len(self.rays),
                size=(self.batch_size,),
                device=self.dataset.all_rgbs.device,
            )
            rays = self.rays[index].to(self.rank)
            rgbs = self.rgbs[index].to(self.rank)
        elif stage in ['val', 'test', 'predict']: 
            rays = self.dataset.all_rays[index].view(-1, 6).to(self.rank)
            rgbs = self.dataset.all_rgbs[index].view(-1, 3).to(self.rank)

        batch.update({
            'rays': rays,
            'rgbs': rgbs,
        })

    def forward(self, batch):
        return self.model(batch["rays"])

    # Refer to https://pytorch-lightning.readthedocs.io/en/1.7.2/common/lightning_module.html#hooks 
    # for hook order.

    def on_train_epoch_start(self) -> None:
        self.dataset = self.trainer.train_dataloader.dataset

    def on_validation_epoch_start(self) -> None:
        self.dataset = self.trainer.val_dataloaders.dataset

    def on_validation_end(self) -> None:
        self.dataset = self.trainer.train_dataloader.dataset

    def on_test_epoch_start(self) -> None:
        self.dataset = self.trainer.test_dataloaders.dataset

    def on_train_batch_start(self, batch, batch_idx) -> None:
        self.preprocess_data(batch, 'train')
        self.update_step(self.global_step)

    def on_validation_batch_start(self, batch, batch_idx) -> None:
        self.preprocess_data(batch, 'val')
        
    def on_test_batch_start(self, batch, batch_idx) -> None:
        self.preprocess_data(batch, 'test')

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = torch.mean((out["rgb_map"] - batch["rgbs"]) ** 2)

        # loss + regularization
        total_loss = loss
        self.log('train/psnr', mse2psnr(loss), prog_bar=True, on_step=True)
        if self.weights_dict["Ortho_reg_weight"] > 0:
            loss_reg = vector_diffs(self.model.DensityField.lines) + vector_diffs(self.model.RadianceField.lines)
            total_loss += self.weights_dict["Ortho_reg_weight"]*loss_reg
            self.log('train/reg', loss_reg, on_step=True)
            # logger.add_scalar('train/reg', loss_reg.detach().item(), self.global_step)
        if self.weights_dict["L1_reg_weight"] > 0:
            loss_reg_L1 = L1_VM(self.model.DensityField.planes, self.model.DensityField.lines)
            total_loss += self.weights_dict["L1_reg_weight"]*loss_reg_L1
            self.log('train/reg_l1', loss_reg_L1, on_step=True)
            # logger.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), self.global_step)
        if self.weights_dict["TV_weight_density"]>0:
            TV_weight_density = self.lr_factor * self.weights_dict["TV_weight_density"]
            loss_tv = TV_weight_density * TVloss(self.model.DensityField.planes)
            total_loss = total_loss + loss_tv
            self.log('train/reg_tv_density', loss_tv, on_step=True)
            # logger.add_scalar('train/reg_tv_density', loss_tv.detach().item(), self.global_step)
        if self.weights_dict["TV_weight_app"]>0:
            TV_weight_app = self.weights_dict["TV_weight_app"] * self.lr_factor
            loss_tv = TV_weight_app * TVloss(self.model.RadianceField.planes)
            total_loss = total_loss + loss_tv
            self.log('train/reg_tv_app', loss_tv, on_step=True)
            # logger.add_scalar('train/reg_tv_app', loss_tv.detach().item(), self.global_step)

        return {
            'loss': total_loss
        }

    def update_step(self, global_step):
        # Update the occupancy grid 
        if global_step in self.update_AlphaMask_list:
            if self.reso_cur[0] * self.reso_cur[1] * self.reso_cur[2]<256**3:# update volume resolution
                self.reso_mask = self.reso_cur
            new_aabb = self.model.OccupancyGrid.update_alpha_volume(self.model.DensityField,
                                                                  self.model.feature2density,
                                                                  tuple(self.reso_mask),
                                                                  self.model.step_size)
            if global_step == self.update_AlphaMask_list[0]:
                self.model.DensityField.shrink(new_aabb, self.model.OccupancyGrid)
                self.model.RadianceField.shrink(new_aabb,self.model.OccupancyGrid)
                self.model.update_stepsize()
                L1_reg_weight = self.config["L1_weight_rest"]
                logger.info_print(f"L1_reg_weight reset: {L1_reg_weight}")

        # Upsample
        if global_step in self.upsamp_list:
            n_voxels = self.N_voxel_list.pop(0)
            self.reso_cur = N_to_reso(n_voxels, self.model.DensityField.aabb)
            self.model.DensityField.upsample(self.reso_cur)
            self.model.RadianceField.upsample(self.reso_cur)
            self.model.update_stepsize()

            if self.config["lr_upsample_reset"]:
                logger.info_print("Reset lr to initial")
                lr_scale = 1 #0.1 ** (global_step / args.n_iters)
            else:
                lr_scale = self.config["lr_decay_target_ratio"] ** (global_step / self.config["n_iters"])
            grad_vars = self.model.get_opt_params(self.config["lr_init"] * lr_scale, self.config["lr_basis"]*lr_scale)
            self.trainer.optimizers = [torch.optim.Adam(grad_vars, betas=(0.9, 0.99))]
        
    def test_step(self, batch, batch_idx):
        out = self(batch)
        W, H = self.dataset.img_wh
        gt_rgb = batch["rgbs"].view(H, W, 3)
        rgb_map = torch.clamp(out["rgb_map"], min=0.0, max=1.0).reshape(H, W, 3).cpu()
        depth_map = out["depth_map"].reshape(H, W).cpu()
        acc_map = torch.clamp(out["acc_map"], min=0.0, max=1.0).reshape(H, W).cpu()
        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), self.dataset.near_far)
        # All metrics in float, not tensor
        psnr = mse2psnr(torch.mean(( rgb_map.to(gt_rgb) - gt_rgb ) ** 2)).item()
        ssim = rgb_ssim(rgb_map.cpu(), gt_rgb.cpu(), 1)
        l_a = rgb_lpips(gt_rgb.cpu().numpy(), rgb_map.numpy(), 'alex', self.rank)
        l_v = rgb_lpips(gt_rgb.cpu().numpy(), rgb_map.numpy(), 'vgg', self.rank)

        logger.write_image(f'it{self.global_step}-test/rgb_{batch["index"][0].item():03d}.png', rgb_map.numpy())
        logger.write_image(f'it{self.global_step}-test/depth_{batch["index"][0].item():03d}.png', depth_map)
        logger.write_image(f'it{self.global_step}-test/alpha_{batch["index"][0].item():03d}.png', acc_map.numpy())

        result = {
            'psnr': psnr,
            'ssim': ssim,
            'l_alex': l_a,
            'l_vgg': l_v,
            'index': batch['index']
        }

        self.test_step_outputs.append(result)


    def on_test_epoch_end(self):
        out = self.test_step_outputs
        if self.trainer.is_global_zero:
            psnr = np.mean([o['psnr'] for o in out])
            ssim = np.mean([o['ssim'] for o in out])
            l_alex = np.mean([o['l_alex'] for o in out])
            l_vgg = np.mean([o['l_vgg'] for o in out])

            logger.info_print(f"test psnr={psnr}")
            logger.info_print(f"test ssim={ssim}")
            logger.info_print(f"test l_alex={l_alex}")
            logger.info_print(f"test l_vgg={l_vgg}")

        self.test_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        W, H = self.dataset.img_wh
        gt_rgb = batch["rgbs"].view(H, W, 3)
        rgb_map = torch.clamp(out["rgb_map"], min=0.0, max=1.0).reshape(H, W, 3).cpu()
        depth_map = out["depth_map"].reshape(H, W).cpu()
        acc_map = torch.clamp(out["acc_map"], min=0.0, max=1.0).reshape(H, W).cpu()
        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), self.dataset.near_far)
        psnr = mse2psnr(torch.mean(( rgb_map.to(gt_rgb) - gt_rgb ) ** 2)).item()
        ssim = rgb_ssim(rgb_map.cpu(), gt_rgb.cpu(), 1)
        l_a = rgb_lpips(gt_rgb.cpu().numpy(), rgb_map.numpy(), 'alex', self.rank)
        l_v = rgb_lpips(gt_rgb.cpu().numpy(), rgb_map.numpy(), 'vgg', self.rank)

        logger.write_image(f'it{self.global_step}-val/rgb_{batch["index"][0].item():03d}.png', rgb_map.numpy())
        logger.write_image(f'it{self.global_step}-val/depth_{batch["index"][0].item():03d}.png', depth_map)
        logger.write_image(f'it{self.global_step}-val/alpha_{batch["index"][0].item():03d}.png', acc_map.numpy())

        result = {
            'psnr': psnr,
            'ssim': ssim,
            'l_alex': l_a,
            'l_vgg': l_v,
            'index': batch['index']
        }

        self.validation_step_outputs.append(result)

    def on_validation_epoch_end(self):
        out = self.validation_step_outputs
        if self.trainer.is_global_zero:
            psnr = np.mean([o['psnr'] for o in out])
            ssim = np.mean([o['ssim'] for o in out])
            l_alex = np.mean([o['l_alex'] for o in out])
            l_vgg = np.mean([o['l_vgg'] for o in out])

            logger.info_print(f"val psnr={psnr}")
            logger.info_print(f"val ssim={ssim}")
            logger.info_print(f"val l_alex={l_alex}")
            logger.info_print(f"val l_vgg={l_vgg}")

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        grad_vars = self.model.get_opt_params(self.config["lr_init"], self.config["lr_basis"])
        optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        return optimizer

    def on_save_checkpoint(self, ckpt):
        ckpt['aabb'] = self.model.DensityField.aabb
        ckpt['reso_cur'] = self.model.DensityField.grid_size
        ckpt['n_samples'] = self.model.n_samples
        ckpt['alpha_volume'] = np.packbits(self.model.OccupancyGrid.alpha_volume.reshape(-1).bool().cpu().numpy())
        ckpt['OccupancyGrid_aabb'] = self.model.OccupancyGrid.aabb
        ckpt['OccupancyGrid_grid_size'] = self.model.OccupancyGrid.grid_size

    def on_load_checkpoint(self, ckpt):
        alpha_volume = ckpt['alpha_volume']
        length = torch.prod(self.model.OccupancyGrid.grid_size, dtype=torch.int)
        self.model.OccupancyGrid.alpha_volume = torch.from_numpy(
            np.unpackbits(alpha_volume)[:length].reshape(self.model.OccupancyGrid.grid_size.tolist()), 
        ).to(self.rank, torch.float)
        self.model.OccupancyGrid.alpha_volume = self.model.OccupancyGrid.alpha_volume.view(1, 1, *self.model.OccupancyGrid.alpha_volume.shape[-3:])
        print(type(self.model.OccupancyGrid.alpha_volume.dtype))
