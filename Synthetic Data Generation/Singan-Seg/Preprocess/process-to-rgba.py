import os
from PIL import Image
import numpy as np
import glob

def process_to_rgba(input_dir, output_dir):
    """
    Converts pairs of images and masks in an input directory to RGBA format and saves them to an output directory.
    
    Parameters:
        - input_dir (str): Path to the directory containing image and mask pairs in .png format in "Preprocess".
        - output_dir (str): Path to the directory where RGBA images will be saved in "Input".
    """
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    
    # Retrieve all .png files and sort them to ensure image-mask pairs are processed together
    paths = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    
    # Loop over image and mask pairs, assuming each image is followed by its mask
    for i in range(0, len(paths), 2):
        image_path = paths[i]       # Path to the MRI image
        mask_path = paths[i + 1]    # Path to the corresponding mask
        
        # Load image and mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Check dimensions of the image and the mask match
        if image.size != mask.size:
            raise ValueError(f"Image and mask sizes do not match for {image_path} and {mask_path}.")
        
        # Convert images to numpy arrays
        image_array = np.array(image)
        mask_array = np.array(mask)
        
        # Stack RGB image and grayscale mask to create an RGBA image
        rgba_image = np.dstack((image_array, mask_array))
        
        # Convert back to PIL RGBA image
        rgba_image_pil = Image.fromarray(rgba_image, 'RGBA')
        
        # Save the RGBA image
        output_filename = os.path.basename(image_path).replace('_image.png', '_RGBA.png')
        output_path = os.path.join(output_dir, output_filename)
        rgba_image_pil.save(output_path)
        
        print(f"Saved 4D image to {output_path}")

# process_to_rgba('/path/to/real_images_and_masks', '/path/to/real_images_rgba') # real data
# process_to_rgba('/path/to/synthetic_images_and_masks', '/path/to/synthetic_images_rgba') # synthetic data

process_to_rgba('/Users/elizabethnemeti/Documents/GitHub/singan-seg/Preprocess/data-pre-RGBA', '/Users/elizabethnemeti/Documents/GitHub/singan-seg/Input/data-RGBA') # real data
process_to_rgba('/path/to/synthetic_images_and_masks', '/path/to/synthetic_images_rgba') # synthetic data
