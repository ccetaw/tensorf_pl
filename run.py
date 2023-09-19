import torch

import json
import sys
import datetime
from tqdm.auto import tqdm

from logger import logger 
from dataLoader import dataset_dict
from utils import *
from models.renderer import NeRFRenderer
from loss import *
from opt import config_parser
from eval import *
from metrics import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reconstruction(args):

    # Init log
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    logger.set_logdir(logfolder)
    logger.set_mode(debug=False) # Set True to print debug information
    logger.write_dict2txt('config.txt', vars(args)) # Save config of this run to logdir

    # Load dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # Model config
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    n_samples = min(args.n_samples, cal_n_samples(reso_cur,args.step_ratio))

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        Renderer = eval(args.model_name)(**kwargs)
        Renderer.load(ckpt)
    else:
        density_field_config = {
            'aabb': aabb,
            'grid_size': reso_cur,
            'n_comp': args.n_lamb_sigma,
            'value_offset': args.density_shift,
            'activation': args.fea2denseAct
        }

        radiance_field_config = {
            'aabb': aabb,
            'grid_size': reso_cur,
            'n_comp': args.n_lamb_sh,
            'value_offset': None,
            'activation': {
                'MLP': args.shadingMode,
                'pos_pe': args.pos_pe,
                'view_pe': args.view_pe,
                'fea_pe': args.fea_pe,
                'featureC': args.featureC
            },
            'dim_4d': args.data_dim_color,
        }

        occupancy_grid_config = {
            'aabb': None,                   # Initially we don't have a occupancy grid
            'grid_size': None,
            'threshold': args.alpha_mask_thre,
        }

        renderer_config = {
            'near_far': near_far,
            'step_ratio': args.step_ratio,
            'distance_scale': args.distance_scale,
            'n_samples': n_samples,
            'ray_march_weight_thres': args.rm_weight_mask_thre,
            'density_field_config': density_field_config,
            'radiance_field_config': radiance_field_config,
            'occupancy_grid_config': occupancy_grid_config
        }

        renderer_config.update({'device': device})
        Renderer = eval(args.model_name)(**renderer_config)

    grad_vars = Renderer.get_opt_params(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    print("lr factor", lr_factor)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))

    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]


    # Init loss weight
    Ortho_reg_weight = args.Ortho_weight
    L1_reg_weight = args.L1_weight_inital
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app

    weights_dict = {
        'ortho_reg_weight': Ortho_reg_weight,
        'L1_reg_weight': L1_reg_weight,
        'TV_weight_density': TV_weight_density,
        'TV_weight_app': TV_weight_app
    }

    print("Initial loss weights ", weights_dict)

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        allrays, allrgbs = Renderer.filter_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    # Train process config
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:] #linear in logrithmic space

    print("Voxel list ", N_voxel_list)

    # Start training
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:

        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx].to(device), allrgbs[ray_idx].to(device)

        rgb_map, depth_map = Renderer(rays_train,
                                      n_samples=n_samples, 
                                      white_bg=white_bg, 
                                      ndc_ray=ndc_ray, 
                                      is_train=True)

        loss = torch.mean((rgb_map - rgb_train) ** 2)

        # loss + regularization
        total_loss = loss
        if Ortho_reg_weight > 0:
            loss_reg = vector_diffs(Renderer.DensityField.lines) + vector_diffs(Renderer.RadianceField.lines)
            total_loss += Ortho_reg_weight*loss_reg
            logger.add_scalar('train/reg', loss_reg.detach().item(), iteration)
        if L1_reg_weight > 0:
            loss_reg_L1 = L1_VM(Renderer.DensityField.planes, Renderer.DensityField.lines)
            total_loss += L1_reg_weight*loss_reg_L1
            logger.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), iteration)
        if TV_weight_density>0:
            TV_weight_density *= lr_factor
            loss_tv = TV_weight_density * TVLoss(Renderer.DensityField.planes)
            total_loss = total_loss + loss_tv
            logger.add_scalar('train/reg_tv_density', loss_tv.detach().item(), iteration)
        if TV_weight_app>0:
            TV_weight_app *= lr_factor
            loss_tv = TV_weight_app * TVLoss(Renderer.RadianceField.planes)
            total_loss = total_loss + loss_tv
            logger.add_scalar('train/reg_tv_app', loss_tv.detach().item(), iteration)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()
        
        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        logger.add_scalar('train/PSNR', PSNRs[-1], iteration)
        logger.add_scalar('train/mse', loss, iteration)

        # lr decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
            )
            PSNRs = []


        # Visualize a few test images every `vis_every` iterations
        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0: 
            PSNRs_test = evaluation(test_dataset, Renderer, save_path=f'ckpt_{iteration}', N_vis=args.N_vis,
                                    n_samples=n_samples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
            logger.add_scalar('test/psnr', np.mean(PSNRs_test), iteration)


        # Update the occupancy grid 
        if iteration in update_AlphaMask_list:
            if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                reso_mask = reso_cur
            new_aabb = Renderer.update_alpha_volume(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                Renderer.DensityField.shrink(new_aabb, Renderer.OccupancyGrid)
                Renderer.RadianceField.shrink(new_aabb,Renderer.OccupancyGrid)
                Renderer.update_stepsize()
                L1_reg_weight = args.L1_weight_rest
                logger.info_print(f"L1_reg_weight reset: {L1_reg_weight}")

            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                allrays, allrgbs = Renderer.filter_rays(allrays, allrgbs)
                trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)

        # Upsample
        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, Renderer.DensityField.aabb)
            n_samples = min(args.n_samples, cal_n_samples(reso_cur,args.step_ratio))
            Renderer.DensityField.upsample(reso_cur)
            Renderer.RadianceField.upsample(reso_cur)
            Renderer.update_stepsize()

            if args.lr_upsample_reset:
                logger.info_print("Reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = Renderer.get_opt_params(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    Renderer.save()

    # Render imgs
    if args.render_train:
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset, Renderer, save_path=f'imgs_train_all', N_vis=args.N_vis,
                                n_samples=n_samples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        PSNRs_test = evaluation(test_dataset, Renderer, save_path=f'imgs_test_all', N_vis=-1,
                                n_samples=n_samples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=True)
        logger.add_scalar('test/psnr_all', np.mean(PSNRs_test), iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    # if args.render_path:
    #     c2ws = test_dataset.render_path
    #     # c2ws = test_dataset.poses
    #     print('========>',c2ws.shape)
    #     os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
    #     evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
    #                             N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(json.dumps(vars(args), indent=4))

    reconstruction(args)
