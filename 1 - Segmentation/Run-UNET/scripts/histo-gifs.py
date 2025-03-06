import os
import cv2
import matplotlib.pyplot as plt
import imageio
import numpy as np
from collections import defaultdict

image_directory = '../data/training_LOO'
output_directory = '../data/histograms_and_gifs'
os.makedirs(output_directory, exist_ok=True) # make directory if not there
subject_files = defaultdict(list) # group files by subject

for filename in os.listdir(image_directory):
    if "_image.png" in filename:  # process only image files not masks
        subject_number = filename.split('_')[1]  # extract subject number
        subject_files[subject_number].append(filename)

# calc RMS contrast
def calculate_rms_contrast(image):
    return np.std(image)

# make the GIFs for each subject
for subject, files in subject_files.items():
    # sort by slice number to ensure correct order
    files.sort(key=lambda x: int(x.split('_')[3]))

    images = []
    max_y_value = 0

    # determine max y value
    for filename in files:
        image_path = os.path.join(image_directory, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        max_y_value = max(max_y_value, max(hist)) # update max y-axis value for consistency

    for filename in files:
        image_path = os.path.join(image_directory, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        rms_contrast = calculate_rms_contrast(image) # Calculate RMS Contrast
        
        # side-by-side plot for better viz
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        # original image on the left
        axs[0].imshow(image, cmap='gray')
        axs[0].axis('off')  # Hide axes
        axs[0].set_title(f"Original Image {filename}\nRMS Contrast: {rms_contrast:.2f}")
        # CNR: {cnr:.2f}
        
        # histogram on the right
        axs[1].fill_between(range(256), hist.flatten(), color='blue')
        axs[1].set_xlim([0, 255])  # ensure x-axis goes up to 255
        axs[1].set_ylim([0, max_y_value * 1.1])  # set y-axis to the max value found
        axs[1].set_xlabel("Pixel Value")
        axs[1].set_ylabel("Frequency")
        axs[1].set_title("Histogram")
        
        # save plots
        plt_path = os.path.join(output_directory, f"{filename}_combined.png")
        plt.savefig(plt_path)
        plt.close()

        # add for gif
        images.append(imageio.imread(plt_path))

    # save as gifs
    gif_path = os.path.join(output_directory, f"{subject}_combined.gif")
    imageio.mimsave(gif_path, images, fps=5)  # change speed here! (FPS)
    
    print(f"GIF saved for subject {subject} at {gif_path}")
