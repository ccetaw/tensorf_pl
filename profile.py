import os 
import datetime
import json
from torch.profiler import profile, ProfilerActivity 
from torch.utils.tensorboard import SummaryWriter

from opt import config_parser
from dataLoader import dataset_dict
from renderer import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = OctreeRender_trilinear_fast

# Replacement of dataloader
class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = [] 

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]

# Load config
args = config_parser()

# print(args)
# print(type(args))
print(json.dumps(vars(args), indent=4))


# Load dataset
dataset = dataset_dict[args.dataset_name]
train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
white_bg = train_dataset.white_bg
near_far = train_dataset.near_far
ndc_ray = args.ndc_ray


# Init model parameters
upsamp_list = args.upsamp_list
update_AlphaMask_list = args.update_AlphaMask_list
n_lamb_sigma = args.n_lamb_sigma
n_lamb_sh = args.n_lamb_sh
aabb = train_dataset.scene_bbox.to(device) 
reso_cur = N_to_reso(args.N_voxel_init, aabb) # current resolution
nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio)) # resolution / step_ratio


# init mode
tensorf = eval(args.model_name)(aabb, reso_cur, device,
            density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
            shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
            pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct)

print(tensorf)

print("Model's state_dict:")
for param_tensor in tensorf.state_dict():
    print(param_tensor, "\t", tensorf.state_dict()[param_tensor].size())

# Process data: filter rays 
allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
if not args.ndc_ray:
    allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)


ray_idx = trainingSampler.nextids()
rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

tvreg = TVLoss()
TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
if args.lr_decay_iters > 0:
    lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
else:
    args.lr_decay_iters = args.n_iters
    lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, tensorf, chunk=args.batch_size,
                            N_samples=nSamples, white_bg=white_bg, ndc_ray=ndc_ray, device=device, is_train=True)
    loss_reg = tensorf.vector_comp_diffs()
    loss_reg_L1 = tensorf.density_L1()
    TV_weight_density *= lr_factor
    loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
    TV_weight_app *= lr_factor
    loss_tv = tensorf.TV_loss_app(tvreg)*TV_weight_app

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=40))


