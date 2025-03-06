"""
The script compares real and synthetic MRI images, focusing specifically on the stomach region, and provides a quantitative assessment of their similarity.
"""
import os
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import dice

def convert_to_rgba(image_path, mask_path, output_path):
    """Combine a synthetic RGB image and its mask into an RGBA image."""
    try:
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        rgba_image = Image.merge("RGBA", (*image.split(), mask))
        rgba_image.save(output_path)
    except Exception as e:
        print(f"Error processing {image_path} and {mask_path}: {e}")

def process_combined_folder(synth_pre_rgba_data_folder, synth_rgba_data_folder):
    """Process a folder containing both images and masks to create RGBA images."""
    os.makedirs(synth_rgba_data_folder, exist_ok=True)
    files = os.listdir(synth_pre_rgba_data_folder)
    
    # Filter images and masks
    images = [f for f in files if f.startswith('image_')]
    masks = [f for f in files if f.startswith('mask_')]
    
    # Create a dictionary for quick lookup of masks
    mask_dict = {f.split('mask_')[1]: f for f in masks}
    
    for image_file in images:
        identifier = image_file.split('image_')[1]
        mask_file = mask_dict.get(identifier)
        
        if mask_file:
            image_path = os.path.join(synth_pre_rgba_data_folder, image_file)
            mask_path = os.path.join(synth_pre_rgba_data_folder, mask_file)
            output_path = os.path.join(synth_rgba_data_folder, f"rgba_{identifier}")
            convert_to_rgba(image_path, mask_path, output_path)
        else:
            print(f"No corresponding mask found for {image_file}, skipping.")

def extract_stomach_roi(image_rgba):
    """Extract the stomach region of interest (ROI) using the alpha channel."""
    mask = image_rgba[..., 3] / 255.0
    roi = image_rgba[..., :3] * mask[:, :, np.newaxis]
    return roi, mask

def compute_ssim_mse(real_roi, synthetic_roi):
    """Compute SSIM and MSE between two ROIs."""
    real_gray = cv2.cvtColor(real_roi.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    synthetic_gray = cv2.cvtColor(synthetic_roi.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    ssim_value = ssim(real_gray, synthetic_gray)
    mse_value = np.mean((real_gray - synthetic_gray) ** 2)
    return ssim_value, mse_value

def compute_dice(real_mask, synthetic_mask):
    """Compute Dice coefficient between two masks."""
    real_bin = (real_mask > 0.5).astype(np.uint8)
    synth_bin = (synthetic_mask > 0.5).astype(np.uint8)
    dice_value = 1 - dice(real_bin.flatten(), synth_bin.flatten())
    return dice_value

def evaluate_synthetic_quality(real_image_rgba, synthetic_image_rgba):
    """Evaluate the quality of synthetic images against real images."""
    real_roi, real_mask = extract_stomach_roi(real_image_rgba)
    synth_roi, synth_mask = extract_stomach_roi(synthetic_image_rgba)
    ssim_value, mse_value = compute_ssim_mse(real_roi, synth_roi)
    dice_value = compute_dice(real_mask, synth_mask)
    return ssim_value, mse_value, dice_value

def evaluate_folder(real_rgba_data_folder, synth_rgba_data_folder):
    """Evaluate all images in the folders and compile results."""
    results = []
    for filename in os.listdir(real_rgba_data_folder):
        real_path = os.path.join(real_rgba_data_folder, filename)
        synth_path = os.path.join(synth_rgba_data_folder, filename)
        if all(os.path.exists(p) for p in [real_path, synth_path]):
            real_image_rgba = np.array(Image.open(real_path))
            synth_image_rgba = np.array(Image.open(synth_path))
            metrics = evaluate_synthetic_quality(real_image_rgba, synth_image_rgba)
            results.append((filename, *metrics))
        else:
            print(f"Missing file for {filename}, skipping.")
    df_results = pd.DataFrame(results, columns=["Image", "SSIM", "MSE", "Dice"]).set_index("Image")
    print(df_results.describe())
    return df_results

if __name__ == "__main__":
    synth_pre_rgba_data_folder = "/Users/elizabethnemeti/Documents/GitHub/singan-seg/Preprocess/synth_data_pre_RGBA"  # contains synthetic data still in RGB format
    real_rgba_data_folder = "/Users/elizabethnemeti/Documents/GitHub/singan-seg/Input/real_data_RGBA"          # contains real data already in RGBA format
    synth_rgba_data_folder = "/Users/elizabethnemeti/Documents/GitHub/singan-seg/Input/synth_data_RGBA"        # contains synthetic data already in RGBA format
    
    # Process the combined folder to create RGBA images
    process_combined_folder(synth_pre_rgba_data_folder, synth_rgba_data_folder)
    
    # Evaluate the synthetic RGBA images against real RGBA images
    evaluate_folder(real_rgba_data_folder, synth_rgba_data_folder)