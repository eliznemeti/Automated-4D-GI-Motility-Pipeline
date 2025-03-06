import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import glob

# Load and separate channels
def plot_image_with_mask(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    # Read the RGBA image
    img = mpimg.imread(image_path)

    # Separate RGB and mask (alpha channel)
    rgb_img = img[:, :, :3]  # Extract the RGB channels
    mask_img = img[:, :, 3]  # Extract the alpha (mask) channel

    # Plot the original RGB image and mask side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display the RGB image
    axs[0].imshow(rgb_img)
    axs[0].set_title('RGB Image')
    axs[0].axis('off')

    # Display the mask (grayscale image)
    axs[1].imshow(mask_img, cmap='gray')
    axs[1].set_title('Grayscale Mask')
    axs[1].axis('off')

    plt.show()

# Iterate through the folders and plot images
root_dir = '/Users/elizabethnemeti/Desktop/FD_029_slice_22_RGBA-4c'
output_dir = '/Users/elizabethnemeti/Documents/GitHub/singan-seg/output_images'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(8):  # Assuming there are 8 folders (0 to 7)
    folder = os.path.join(root_dir, str(i))
    original_img_path = os.path.join(folder, 'real_scale.png')
    synthetic_img_path = os.path.join(folder, 'fake_sample.png')

    print(f"Original image path: {original_img_path}")
    print(f"Synthetic image path: {synthetic_img_path}")

    # Plot the original and synthetic images
    plot_image_with_mask(original_img_path)
    plot_image_with_mask(synthetic_img_path)

# # Function to load images and masks
# def load_images_and_masks_v2(filenames):
#     images = []
#     masks = []
#     real  = []
#     for filename in sorted(filenames):
#         if 'slice' in filename and 'mask' not in filename:
#             real_image = np.array(Image.open(filename))
#         if 'mask' in filename and 'slice' in filename:
#             real_mask = np.array(Image.open(filename))
#         if 'mask' in filename and 'slice' not in filename:
#             masks.append(np.array(Image.open(filename)))
#         else:
#             images.append(np.array(Image.open(filename)))
#     return images, masks, real_image, real_mask
# case_paths = glob.glob('/home/syurtseven/gsoc-2024/external/singan-seg/plot_figures/case2*.png')
# images, masks, real_image, real_mask = load_images_and_masks_v2(case_paths)
# # Separate real and generated images/masks
# generated_images_v2 = images
# generated_masks_v2 = masks
# # Create the subplot structure
# fig, axes = plt.subplots(nrows=2, ncols=9, figsize=(18, 6))
# # Plot the real image and its mask
# axes[0, 0].imshow(real_image)
# axes[0, 0].axis('off')
# axes[1, 0].imshow(real_mask, cmap='gray')
# axes[1, 0].axis('off')
# # Plot the generated images and their masks
# for i in range(8):
#     axes[0, i + 1].imshow(generated_images_v2[i])
#     axes[0, i + 1].axis('off')
#     axes[1, i + 1].imshow(generated_masks_v2[i], cmap='gray')
#     axes[1, i + 1].axis('off')
# # Add titles
# axes[0, 0].set_title('Real images\nand masks', fontsize=12)
# axes[0, 1].set_title('Generated images and masks', fontsize=12, loc='left', pad=20)
# plt.tight_layout()
# plt.show()