import torch
from torch.utils.data import DataLoader

import json
import os
import sys
import datetime

from logger import logger 
from dataLoader import dataset_dict
from opt import config_parser
from system import TensoRF
from utils import N_to_reso, cal_n_samples

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Print config information
    args = config_parser()
    print(json.dumps(vars(args), indent=4))

    # Init log
    if args.train:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
        logger.set_logdir(logfolder)
        logger.set_mode(debug=False) # Set True to print debug information
        logger.write_dict2txt('config.txt', vars(args).copy()) # Save config of this run to logdir
    else:
        logger.set_mode(debug=False)
        logfolder = args.config.replace(os.path.basename(args.config), '')
        logger.set_logdir(logfolder)

    # Load dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
    train_dataloader = DataLoader(train_dataset, num_workers=8, batch_size=1, pin_memory=True)
    val_dataset = dataset(args.datadir, split='val', downsample=args.downsample_train, is_stack=True)
    val_dataloader = DataLoader(val_dataset, num_workers=8, batch_size=1, pin_memory=True)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    test_dataloader = DataLoader(test_dataset, num_workers=8, batch_size=1, pin_memory=True)

    config = vars(args)
    config.update({
        'white_bg': train_dataset.white_bg,
        'near_far': train_dataset.near_far,
        'device': device,
    })

    if args.resume:
        ckpt = torch.load(args.resume)
        aabb = ckpt['aabb']
        reso_cur = ckpt['reso_cur']
        n_samples = ckpt['n_samples']
        OccupancyGrid_aabb = ckpt['OccupancyGrid_aabb']
        OccupancyGrid_grid_size = ckpt['OccupancyGrid_grid_size']
        del ckpt
    else:
        aabb = train_dataset.scene_bbox.to(device)
        reso_cur = N_to_reso(config["N_voxel_init"], aabb)
        n_samples = min(config["n_samples"], cal_n_samples(reso_cur, config["step_ratio"]))
        OccupancyGrid_aabb = None
        OccupancyGrid_grid_size = None

    config.update({
        'aabb': aabb,
        'reso_cur': reso_cur,
        'n_samples': n_samples,
        'OccupancyGrid_aabb': OccupancyGrid_aabb,
        'OccupancyGrid_grid_size': OccupancyGrid_grid_size
    })

    system = TensoRF(config)

    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DDPStrategy
    from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

    pl.seed_everything(42)

    if sys.platform == 'win32':
        # does not support multi-gpu on windows
        strategy = 'dp'
        assert n_gpus == 1
    else:
        strategy = DDPStrategy(find_unused_parameters=True)

    callbacks = []
    if args.train:
        callbacks += [
            ModelCheckpoint(
                dirpath=os.path.join(logfolder, 'ckpt'),
                every_n_train_steps=args.n_iters,
                save_top_k=-1
            ),
            TQDMProgressBar(refresh_rate=1)
        ]

    trainer = Trainer(
        devices=1,
        accelerator='gpu',
        callbacks=callbacks,
        max_steps=args.n_iters,
        enable_progress_bar=True,
        strategy=strategy,
        precision=32,
        limit_val_batches=args.N_vis,
        # val_check_interval=args.vis_every,
        default_root_dir=logfolder,
        num_sanity_val_steps=0,
        limit_train_batches=args.n_iters
    )

    if args.train:
        if args.resume is not None:
            trainer.fit(system, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=args.resume)
        else:
            trainer.fit(system, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        trainer.test(system, test_dataloader)
    elif args.test and args.resume is not None:
        trainer.test(system, test_dataloader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
