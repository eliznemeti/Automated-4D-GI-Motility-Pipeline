import os
import scipy.io
import numpy as np
import cv2
import re

# Define the directory containing the .mat files
data_dir = '/Users/elizabethnemeti/Desktop/roberta_3d_data'

# Define the output directory for the .png files
output_dir = os.path.join(data_dir, 'converted_png_files')

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all .mat files in the directory
mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]

# Process each .mat file
for mat_file in mat_files:
    mat_path = os.path.join(data_dir, mat_file)
    
    print(f"\nProcessing {mat_file}...")
    
    # Load the .mat file
    try:
        mat_contents = scipy.io.loadmat(mat_path)
    except Exception as e:
        print(f"Error loading {mat_file}: {e}")
        break  # Stop execution
    
    # Remove the .mat extension
    filename_without_ext = os.path.splitext(mat_file)[0]
    
    # Use regex to extract parts of the filename
    match = re.match(r'(FD_\d+)_(image|mask)_time_(\d+)', filename_without_ext)
    if not match:
        print(f"Filename {mat_file} does not match expected pattern. Stopping execution.")
        break  # Stop execution

    fd_id, img_or_mask, time_number = match.groups()

    # Create the new filename
    new_filename = f"{fd_id}_time_{time_number}_{img_or_mask}.png"

    output_path = os.path.join(output_dir, new_filename)

    # Extract the data based on whether it's an image or a mask
    if img_or_mask == 'image':
        variable_name = 'image_time'
    elif img_or_mask == 'mask':
        variable_name = 'mask_time'
    else:
        print(f"Unknown type '{img_or_mask}' in {mat_file}. Stopping execution.")
        break  # Stop execution

    if variable_name in mat_contents:
        image_data = mat_contents[variable_name]
    else:
        print(f"Variable '{variable_name}' not found in {mat_file}. Stopping execution.")
        print(f"Available variables: {list(mat_contents.keys())}")
        break  # Stop execution

    # image_data should be a 3D array of size (192, 156, 72)
    # For visualization, we can take the middle slice along the z-axis (axis=2)

    if image_data.ndim == 3:
        middle_index = image_data.shape[2] // 2
        image_slice = image_data[:, :, middle_index]
    elif image_data.ndim == 2:
        image_slice = image_data
    else:
        print(f"Unsupported image dimensions in {mat_file}: {image_data.shape}. Stopping execution.")
        break  # Stop execution

    # Normalize image data to 0-255
    image_slice = image_slice.astype(np.float32)
    min_val = image_slice.min()
    max_val = image_slice.max()
    if max_val - min_val > 0:
        image_slice = (image_slice - min_val) / (max_val - min_val)
    else:
        image_slice = np.zeros_like(image_slice)
    image_slice = (image_slice * 255).astype(np.uint8)

    # Save the image data as a PNG file
    try:
        cv2.imwrite(output_path, image_slice)
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
        break  # Stop execution

    print(f"Converted {mat_file} to {new_filename}")

print("Conversion completed.")
