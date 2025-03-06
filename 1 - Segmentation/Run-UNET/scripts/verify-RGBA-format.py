import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def check_rgba_file(image_rgba, show_image=True):
    if image_rgba.shape[2] != 4:
        raise ValueError("Image does not have 4 channels. Please check your RGBA file.")

    rgb = image_rgba[..., :3]    # RGB channels
    alpha = image_rgba[..., 3]   # Alpha channel (stomach mask)

    print(f"Image dimensions (HxWxC): {image_rgba.shape}")
    
    print("RGB channel statistics:")
    print(f"  Mean: {np.mean(rgb)}")
    print(f"  Max: {np.max(rgb)}")
    print(f"  Min: {np.min(rgb)}")

    print("Alpha channel (mask) statistics:")
    print(f"  Mean: {np.mean(alpha)}")
    print(f"  Max: {np.max(alpha)}")
    print(f"  Min: {np.min(alpha)}")

    if show_image:
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        
        # Show RGB channels combined
        axs[0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        axs[0].set_title("RGB")
        
        axs[1].imshow(rgb[..., 0], cmap="gray")
        axs[1].set_title("Red Channel")
        
        axs[2].imshow(rgb[..., 1], cmap="gray")
        axs[2].set_title("Green Channel")
        
        axs[3].imshow(alpha, cmap="gray")
        axs[3].set_title("Alpha Channel (Stomach Mask)")
        
        plt.show()

rgba_dir = '/Users/elizabethnemeti/Documents/GitHub/singan-seg/Input/synth_data_RGBA'

for filename in os.listdir(rgba_dir):
    if filename.endswith(".png"):
        file_path = os.path.join(rgba_dir, filename)
        image_rgba = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        print(f"Checking file: {filename}")
        check_rgba_file(image_rgba)
        print("\n" + "="*40 + "\n")