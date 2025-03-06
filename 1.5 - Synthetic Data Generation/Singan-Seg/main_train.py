"""
This script trains the SinGAN model on a specified input image.

Usage:
    python main_train.py --input_name <image_name> [--input_dir <directory>] [--mode <mode>] [--gpu_id <id>]

Arguments:
    --input_name : str
        Name of the input image file (required).
    --input_dir : str, optional
        Directory containing the input image (default: 'Input/data-RGBA').
    --mode : str, optional
        Operation mode (default: 'train').
    --gpu_id : str, optional
        GPU ID to use for training.

Functionality:
    - Parses command-line arguments to configure the training process.
    - Loads and preprocesses the input image.
    - Adjusts image scales to fit the model's requirements.
    - Initiates the training process, building a pyramid of generative models.
    - Optionally generates samples based on the trained model.
"""

# Required Packages
from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
import torch
#

# Code Task:
# setup for training a SinGAN model 
# handles configuration, training, and generation processes of the SinGAN model

if __name__ == '__main__': # makes sure script only runs the code if executed as main program
    parser = get_arguments() # initialize a parser for command-line arguments
    parser.add_argument('--input_dir', help='input image dir', default='Input/data-RGBA')
    parser.add_argument('--input_name', help='input image name', required=True) 
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--gpu_id', help='GPU ID to train')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:{}".format(opt.gpu_id))
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, NoiseAmp)
        # Call SinGAN_generate with real image path for naming
        #SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt, real_image_path=opt.input_name)
        SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)