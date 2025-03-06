import os
import imageio
from collections import defaultdict

# Directories
image_directory = '../reports/seg/testinggrid'
output_directory = '../testinggrid_gifs'
slice_over_epochs_dir = os.path.join(output_directory, "see_slice_X_over_epochs")
epoch_for_slices_dir = os.path.join(output_directory, "see_epoch_X_over_slices")

os.makedirs(slice_over_epochs_dir, exist_ok=True)
os.makedirs(epoch_for_slices_dir, exist_ok=True)

# Collect images by epoch and slice number
epoch_images = defaultdict(list)
slice_images = defaultdict(list)

for filename in os.listdir(image_directory):
    if filename.endswith(".jpg") and "loss" not in filename:  # Ensure only images are processed, and exclude loss-related files
        epoch_number = filename.split('_')[1]  # Extract epoch number
        slice_number = filename.split('_')[-1].split('.')[0].replace('slice', '')  # Extract slice number
        epoch_images[epoch_number].append(filename)
        slice_images[slice_number].append(filename)

# Generate GIFs for each epoch (original functionality) and save in "epoch_for_slices_results"
for epoch, files in epoch_images.items():
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0].replace('slice', '')))  # Sort by slice number
    
    images = []
    for filename in files:
        image_path = os.path.join(image_directory, filename)
        images.append(imageio.imread(image_path))

    # Save the GIF for the current epoch in "epoch_for_slices_results"
    gif_path = os.path.join(epoch_for_slices_dir, f"epoch_{epoch}.gif")
    imageio.mimsave(gif_path, images, fps=5)  # Adjust FPS for speed
    
    print(f"GIF saved for epoch {epoch} at {gif_path}")

# Generate GIFs for each slice across epochs (new functionality) and save in "slice_over_epochs_results"
for slice_num, files in slice_images.items():
    # Sort by epoch number, assuming the epoch number is the second element in the filename
    files.sort(key=lambda x: int(x.split('_')[1]))
    
    images = []
    for filename in files:
        image_path = os.path.join(image_directory, filename)
        images.append(imageio.imread(image_path))

    # Save the GIF for the current slice in "slice_over_epochs_results"
    gif_path = os.path.join(slice_over_epochs_dir, f"slice_{slice_num}.gif")
    imageio.mimsave(gif_path, images, fps=5)  # Adjust FPS for speed
    
    print(f"GIF saved for slice {slice_num} at {gif_path}")

# Optionally, create a composite GIF with all slices across epochs and save in "slice_over_epochs_results"
composite_gif_path = os.path.join(slice_over_epochs_dir, "composite_slices.gif")
composite_images = []

for slice_num in sorted(slice_images.keys(), key=lambda x: int(x)):
    slice_gif_path = os.path.join(slice_over_epochs_dir, f"slice_{slice_num}.gif")
    composite_images.append(imageio.imread(slice_gif_path))

# Save the composite GIF for all slices across epochs
imageio.mimsave(composite_gif_path, composite_images, fps=10)  # Slower FPS for viewing

print(f"Composite GIF saved at {composite_gif_path}")
