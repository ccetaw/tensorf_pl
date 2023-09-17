import torch
from tqdm.auto import tqdm
import sys

from utils import *
from logger import logger


@torch.no_grad()
def evaluation(test_dataset, renderer, save_path='', N_vis=5, n_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    """
    Evaluate test dataset. 
    """
    PSNRs, ssims, l_alex, l_vgg= [], [], [], []

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):
        # Render one image for one iter

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        # Split into chunks 
        rgb_map, depth_map = renderer.chunked_render(rays, chunk=4096, white_bg=white_bg, ndc_ray=ndc_ray, is_train=False, n_samples=n_samples)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        # Save imgaes
        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        logger.write_image(f'{save_path}/rgb_{idx:03d}.png', rgb_map)
        rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        logger.write_image(f'{save_path}/rgbd_{idx:03d}.png', rgb_map)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        ckpt_metrics = {
            'psnr': psnr
        }
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            ckpt_metrics.update({
                'ssim': ssim,
                'lpips_alex': l_a,
                'lpips_vgg': l_v
            })
        logger.write_dict2txt(f'{save_path}/metrics.txt', ckpt_metrics)

    return PSNRs
