#!/bin/sh
#Shell script to generate random samples from a pre-trained SinGAN model using default size.

python random_samples.py --input_name polyp_4_channel_test_1.png --mode random_samples --gen_start_scale 0 --nc_z 4 --nc_im 4
