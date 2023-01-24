import os
import sys
import time
from argparse import Namespace

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm


import os

import clip
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# import criteria.clip_loss as clip_loss
# from criteria import id_loss
# from editings.styleclip_mapper.datasets.latents_dataset import LatentsDataset
# from editings.styleclip_mapper.styleclip_mapper import StyleCLIPMapper
# from utils import train_utils
# from training.ranger import Ranger

sys.path.append(".")
sys.path.append("..")

from editings.styleclip_mapper.datasets.latents_dataset import LatentsDataset

from editings.styleclip_mapper.options.test_options import TestOptions
from editings.styleclip_mapper.styleclip_mapper import StyleCLIPMapper
from PIL import Image

def tensor2im(var):
	# var shape: (3, H, W)
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes

def run(test_opts):
    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    os.makedirs(out_path_results, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)

    net = StyleCLIPMapper(opts)
    net.eval()
    net.cuda()

    test_latents = torch.load(opts.latents_test_path)
    if opts.fourier_features_transforms_path:
        transforms = np.load(opts.fourier_features_transforms_path, allow_pickle=True)
    else:
        transforms = None
    dataset = LatentsDataset(root='/content/drive/MyDrive/diffusion_based/celeb/CelebAMask-HQ/CelebA-HQ-img', opts=opts)

    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    global_time = []
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():

            x = input_batch
            x = x.to('cuda')
            print(x.shape)
            w = get_latents(net.net , x)

            w_hat = w + 0.1 * net.mapper(w)
            
            with torch.no_grad():
                x_hat , _ = net.net.decoder([w_hat] , None, input_is_latent=True, randomize_noise=False, return_latents=True)
                img_edit = torch.nn.functional.interpolate(torch.clamp(x_hat, -1., 1.), size=(256,256) , mode='bilinear')
                res_ = x - img_edit
                
                _ , res_align  = net.net.grid_align(torch.cat((res_, img_edit), 1))
                res_align = res_align + torch.cat((res_, img_edit  ), 1)      

                conditions = net.net.residue(res_align)           

                x_hat, _ = net.net.decoder([w_hat],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)
    
            x_hat = torch.nn.functional.interpolate(torch.clamp(x_hat, -1., 1.), size=(256,256) , mode='bilinear')


            ############

        for i in range(opts.test_batch_size):
            im_path = str(global_i).zfill(5)
            if test_opts.couple_outputs:
                couple_output = torch.cat([result_batch[2][i].unsqueeze(0), result_batch[0][i].unsqueeze(0)])
                torchvision.utils.save_image(couple_output, os.path.join(out_path_results, f"{im_path}.jpg"), normalize=True, range=(-1, 1))
            else:
                tensor2im(x_hat[0]).save(os.path.join(out_path_results, f"{im_path}.jpg"))
                # torchvision.utils.save_image(tensor2im(x_hat[0][i]), os.path.join(out_path_results, f"{im_path}.jpg"), normalize=True, range=(-1, 1))
            # torch.save(x_hat[1][i].detach().cpu(), os.path.join(out_path_results, f"latent_{im_path}.pt"))

            global_i += 1

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)


def run_on_batch(inputs, transform, net, couple_outputs=False):
    w = inputs
    with torch.no_grad():
        w_hat = w + 0.1 * net.mapper(w)
        if transform is not None:
            net.decoder.synthesis.input.transform = transform
        x_hat = net.decoder.synthesis(w_hat)
        result_batch = (x_hat, w_hat)
        if couple_outputs:
            x = net.decoder.synthesis(w)
            result_batch = (x_hat, w_hat, x)
    return result_batch



if __name__ == '__main__':
    test_opts = TestOptions().parse()
    run(test_opts)