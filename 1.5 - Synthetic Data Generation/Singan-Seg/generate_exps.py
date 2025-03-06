"""
This script automates the training process for multiple images by sequentially invoking 'main_train.py' on each image in a specified directory.

Usage:
    python generate_exps.py

Functionality:
    - Scans the 'Input/post-RGBA-4c/' directory for PNG images.
    - For each image found, constructs a command to run 'main_train.py' with appropriate arguments.
    - Executes the command using Python's 'subprocess' module, effectively training the SinGAN model on each image in the directory.
    - Handles exceptions and logs the status of each training process.
"""

import os
import subprocess
import glob

def sample_gan(n_samples=1):

    images = glob.glob("/Users/elizabethnemeti/Documents/GitHub/singan-seg/Input/post-RGBA-4c/*.png") # use the concatenated RGBA-4c images

    for img in images:
        
        img_name = img.split('/')[-1]
        python_command = f"python main_train.py --input_name {img_name} --nc_z 4 --nc_im 4 --gpu_id 0"
        
        try:
            print(f"Running command: {python_command}")
            result = subprocess.run(python_command, shell=True, check=True)
            print(f"Command completed successfully for {img_name}\n")
        
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing {img_name}: {e}")
            break 
               
sample_gan()