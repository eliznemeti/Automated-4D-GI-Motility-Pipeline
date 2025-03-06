"""
Script containing functions to manipulate SinGAN-generated images, including GIF creation and sample generation.

Key Functions and Parameters:
    - `generate_gif(...)`: Generates a GIF by iterating through the trained SinGAN scales.
    - `SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, in_s=None, scale_v=1, scale_h=1, n=0, gen_start_scale=0, num_samples=5)`:
        - `Gs`: List of trained generator models for each scale.
        - `Zs`: List of noise tensors corresponding to each scale.
        - `reals`: List of real images at each scale.
        - `NoiseAmp`: List of noise amplitude values for each scale.
        - `opt`: Configuration object containing various settings.
        - `in_s` (optional): Initial input tensor; defaults to `None`.
        - `scale_v`: Vertical scaling factor; defaults to `1`.
        - `scale_h`: Horizontal scaling factor; defaults to `1`.
        - `n`: Starting scale index; defaults to `0`.
        - `gen_start_scale`: Scale to begin generation; defaults to `0`.
        - `num_samples`: Number of synthetic samples to generate`.
    - To adjust the number of synthetic pairs generated, modify the `num_samples` parameter.
    - To change output filenames, update the `plt.imsave` function calls, e.g., `plt.imsave('%s/%d_mask.png', ...)`.

Usage:
    This script is typically imported by other scripts or used to generate new visual outputs based on trained SinGAN models.
"""

from __future__ import print_function
import SinGAN.functions
import SinGAN.models
import argparse
import os
import random
from SinGAN.imresize import imresize
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from skimage import io as img
import numpy as np
from skimage import color
import math
import imageio
import matplotlib.pyplot as plt
from SinGAN.training import *
from config import get_arguments

def generate_gif(Gs,Zs,reals,NoiseAmp,opt,alpha=0.1,beta=0.9,start_scale=2,fps=10):

    in_s = torch.full(Zs[0].shape, 0, device=opt.device, dtype=torch.bool)
    images_cur = []
    count = 0

    for G,Z_opt,noise_amp,real in zip(Gs,Zs,NoiseAmp,reals):
        pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        nzx = Z_opt.shape[2]
        nzy = Z_opt.shape[3]
        #pad_noise = 0
        #m_noise = nn.ZeroPad2d(int(pad_noise))
        m_image = nn.ZeroPad2d(int(pad_image))
        images_prev = images_cur
        images_cur = []
        if count == 0:
            z_rand = functions.generate_noise([1,nzx,nzy], device=opt.device)
            z_rand = z_rand.expand(1,opt.nc_z,Z_opt.shape[2],Z_opt.shape[3])
            z_prev1 = 0.95*Z_opt +0.05*z_rand
            z_prev2 = Z_opt
        else:
            z_prev1 = 0.95*Z_opt + 0.05*functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device)
            z_prev2 = Z_opt

        for i in range(0,100,1):
            if count == 0:
                z_rand = functions.generate_noise([1,nzx,nzy], device=opt.device)
                z_rand = z_rand.expand(1,opt.nc_z,Z_opt.shape[2],Z_opt.shape[3])
                diff_curr = beta*(z_prev1-z_prev2)+(1-beta)*z_rand
            else:
                diff_curr = beta*(z_prev1-z_prev2)+(1-beta)*(functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device))

            z_curr = alpha*Z_opt+(1-alpha)*(z_prev1+diff_curr)
            z_prev2 = z_prev1
            z_prev1 = z_curr

            if images_prev == []:
                I_prev = in_s
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev, 1 / opt.scale_factor, opt)
                I_prev = I_prev[:, :, 0:real.shape[2], 0:real.shape[3]]
                #I_prev = functions.upsampling(I_prev,reals[count].shape[2],reals[count].shape[3])
                I_prev = m_image(I_prev)
            if count < start_scale:
                z_curr = Z_opt

            z_in = noise_amp*z_curr+I_prev
            I_curr = G(z_in.detach(),I_prev)

            if (count == len(Gs)-1):
                I_curr = functions.denorm(I_curr).detach()
                I_curr = I_curr[0,:,:,:].cpu().numpy()
                I_curr = I_curr.transpose(1, 2, 0)*255
                I_curr = I_curr.astype(np.uint8)

            images_cur.append(I_curr)
        count += 1
    dir2save = functions.generate_dir2save(opt)
    try:
        os.makedirs('%s/start_scale=%d' % (dir2save,start_scale) )
    except OSError:
        pass
    imageio.mimsave('%s/start_scale=%d/alpha=%f_beta=%f.gif' % (dir2save,start_scale,alpha,beta),images_cur,fps=fps)
    del images_cur

# Set parameters here
def SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, real_image_path=None, in_s=None, scale_v=1, scale_h=1, n=0, gen_start_scale=0, num_samples=1):
    print("SinGAN_generate is called with real_image_path:", real_image_path)
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device, dtype=torch.bool)

    # Extract the base name from the real image filename
    real_image_base_name = os.path.splitext(os.path.basename(real_image_path))[0].replace('_image', '')

    images_cur = []
    for G, Z_opt, noise_amp in zip(Gs, Zs, NoiseAmp):
        pad1 = ((opt.ker_size - 1) * opt.num_layer) / 2
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2] - pad1 * 2) * scale_v
        nzy = (Z_opt.shape[3] - pad1 * 2) * scale_h

        images_prev = images_cur
        images_cur = []

        for i in range(num_samples):
            if n == 0:
                z_curr = functions.generate_noise([1, nzx, nzy], device=opt.device)
                z_curr = z_curr.expand(1, opt.nc_z, z_curr.shape[2], z_curr.shape[3])
                z_curr = m(z_curr)
            else:
                z_curr = functions.generate_noise([opt.nc_z, nzx, nzy], device=opt.device)
                z_curr = m(z_curr)

            if not images_prev:
                I_prev = m(in_s)
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev, 1 / opt.scale_factor, opt)
                I_prev = I_prev[:, :, :round(scale_v * reals[n].shape[2]), :round(scale_h * reals[n].shape[3])]
                I_prev = m(I_prev)
                I_prev = I_prev[:, :, :z_curr.shape[2], :z_curr.shape[3]]
                I_prev = functions.upsampling(I_prev, z_curr.shape[2], z_curr.shape[3])

            if n < gen_start_scale:
                z_curr = Z_opt

            z_in = noise_amp * z_curr + I_prev
            I_curr = G(z_in.detach(), I_prev)

            # Save only at the last scale
            if n == len(reals) - 1:
                dir2save = functions.generate_dir2save(opt)
                os.makedirs(dir2save, exist_ok=True)

                # Define filenames based on the base name of the real image
                image_filename = f'{real_image_base_name}_synth_image.png'
                mask_filename = f'{real_image_base_name}_synth_mask.png'

                # Add the print statements here to check the filenames
                print(f"Saving image to: {os.path.join(dir2save, image_filename)}")
                print(f"Saving mask to: {os.path.join(dir2save, mask_filename)}")


                # Save the synthetic image and mask
                plt.imsave(os.path.join(dir2save, image_filename), functions.convert_image_np(I_curr.detach())[:, :, :3], vmin=0, vmax=1)
                plt.imsave(os.path.join(dir2save, mask_filename), functions.convert_image_np(I_curr.detach())[:, :, 3], vmin=0, vmax=1)

            images_cur.append(I_curr)
        n += 1
    return I_curr.detach()