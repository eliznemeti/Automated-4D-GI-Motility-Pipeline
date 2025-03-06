'''
Python script to move the synthetic images and masks from the nested folders into the main folder /synthetic_data_rgb for further processing.
Renames files to suit rgba converter script requirements.
'''

import os
import shutil
from PIL import Image  # Ensure Pillow is installed

source_dir = '/Users/elizabethnemeti/Documents/GitHub/Singan-Seg-GI/Output/RandomSamples'
dest_dir = '/Users/elizabethnemeti/Documents/GitHub/Singan-Seg-GI/Preprocess/RandomSamples_ready'
os.makedirs(dest_dir, exist_ok=True)

for subdir in os.listdir(source_dir):
    subdir_path = os.path.join(source_dir, subdir)
    if os.path.isdir(subdir_path):
        print(f"Processing subdirectory: {subdir_path}")
        gen_scale_dir = os.path.join(subdir_path, 'gen_start_scale=0')  # img/mask data stored here
        if os.path.isdir(gen_scale_dir):
            print(f"  Found 'gen_start_scale=0': {gen_scale_dir}")
            files = os.listdir(gen_scale_dir)
            if files:
                print(f"  Files in directory: {files}")
            else:
                print(f"  No files found in {gen_scale_dir}")
            for filename in files:
                file_path = os.path.join(gen_scale_dir, filename)
                if os.path.isfile(file_path):
                    print(f"  Processing file: {file_path}")
                    if 'mask' in filename.lower():
                        new_filename = os.path.splitext(filename)[0] + '_mask.png'
                    else:
                        new_filename = os.path.splitext(filename)[0] + '_image.png'

                    dest_file_path = os.path.join(dest_dir, new_filename)

                    counter = 1
                    original_new_filename = new_filename
                    while os.path.exists(dest_file_path):
                        base, ext = os.path.splitext(original_new_filename)
                        new_filename = f"{base}_{counter}{ext}"
                        dest_file_path = os.path.join(dest_dir, new_filename)
                        counter += 1

                    try:
                        with Image.open(file_path) as img:
                            grayscale_img = img.convert('L')
                            grayscale_img.save(dest_file_path)
                            print(f"Converted, moved, and renamed: {file_path} -> {dest_file_path}")
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
