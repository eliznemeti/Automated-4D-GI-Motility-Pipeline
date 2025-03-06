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

def process_folders(real_folder, synthetic_folder, masks_folder, output_folder):
    """Process folders to convert synthetic images and masks to RGBA format."""
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(synthetic_folder):
        real_path = os.path.join(real_folder, filename)
        synth_path = os.path.join(synthetic_folder, filename)
        mask_path = os.path.join(masks_folder, filename)
        output_path = os.path.join(output_folder, filename)
        if all(os.path.exists(p) for p in [real_path, synth_path, mask_path]):
            convert_to_rgba(synth_path, mask_path, output_path)
        else:
            print(f"Missing file for {filename}, skipping.")

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

def evaluate_folder(real_folder, synthetic_rgba_folder):
    """Evaluate all images in the folders and compile results."""
    results = []
    for filename in os.listdir(real_folder):
        real_path = os.path.join(real_folder, filename)
        synth_path = os.path.join(synthetic_rgba_folder, filename)
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
    real_folder = "/path/to/real_data_rgba"         # converted to rgba, combines real images + real masks
    output_folder = "/path/to/synthetic_data_rgba"  # converted to rgba, combines synthetic images + synthetic masks
    synthetic_folder = "/path/to/synthetic_images"  # typically rgb
    masks_folder = "/path/to/real_masks"            # typically grayscale
    process_folders(real_folder, synthetic_folder, masks_folder, output_folder)
    evaluate_folder(real_folder, output_folder)