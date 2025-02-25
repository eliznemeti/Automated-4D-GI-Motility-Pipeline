Author: Elizabeth N.

import os
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm

# Path to the folder containing the 2D slices
input_folder = '/Users/2D-roberta-data'
output_folder = '/Users/Reconstructed-3D-original'
os.makedirs(output_folder, exist_ok=True)

# Function to parse the file name and extract subject, time, and slice info
def parse_filename(filename):
    parts = filename.split('_')
    
    try:
        subject = f"{parts[0]}_{parts[1]}"  # Combine 'FD' and '031'
        time_point = int(parts[3])  # '86' is in parts[3]
        slice_idx = int(parts[5])  # '61' is in parts[5]
    except (IndexError, ValueError) as e:
        raise ValueError(f"Error parsing time or slice index in file: {filename}") from e

    return subject, time_point, slice_idx

# Filter files to include only masks
files = [f for f in os.listdir(input_folder) if f.endswith('_mask.png')]
data_dict = {}

# Organize files by subject and time point
for file in files:

    subject, time_point, slice_idx = parse_filename(file)
    if subject not in data_dict:
        data_dict[subject] = {}
    if time_point not in data_dict[subject]:
        data_dict[subject][time_point] = {}
    data_dict[subject][time_point][slice_idx] = file

# Reconstruct 3D volumes for each subject and time point
for subject, time_data in tqdm(data_dict.items(), desc="Reconstructing subjects"):
    subject_output_folder = os.path.join(output_folder, subject)
    os.makedirs(subject_output_folder, exist_ok=True)

    for time_point, slices in time_data.items():
        # Determine the number of slices
        max_slice_idx = max(slices.keys())
        
        # Load slices into a 3D numpy array
        slice_shape = None
        volume = []

        for slice_idx in range(1, max_slice_idx + 1):
            if slice_idx in slices:
                slice_path = os.path.join(input_folder, slices[slice_idx])
                slice_image = np.array(Image.open(slice_path))  # Load PNG as numpy array

                if slice_shape is None:
                    slice_shape = slice_image.shape
                volume.append(slice_image)
            else:
                # Fill missing slices with zeros
                if slice_shape is not None:
                    volume.append(np.zeros(slice_shape))

        # Stack slices into a 3D volume
        volume = np.stack(volume, axis=-1)  # Stack along the z-axis
        # Normalize volume to the range [0, 255]
        volume = volume.astype(np.float32)  # Ensure float for normalization
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))  # Normalize to [0, 1]
        volume = (volume * 255).astype(np.uint8)  # Scale to [0, 255] as uint8
        volume_reoriented = np.transpose(volume, (1, 0, 2))

        # Save as a .nii.gz file
        output_path = os.path.join(subject_output_folder, f"{subject}_time_{time_point}_reconstructed.nii.gz")
        spacing = (1.0, 1.0, 1.0)  # Example: Isotropic spacing of 1 mm
        nii = nib.Nifti1Image(volume, np.eye(4))
        nii.header.set_zooms(spacing)  # Set voxel spacing
        nib.save(nii, output_path)
        
        print(f"Reconstructed volume shape: {volume.shape}")
        print(f"Intensity range: {volume.min()} to {volume.max()}")