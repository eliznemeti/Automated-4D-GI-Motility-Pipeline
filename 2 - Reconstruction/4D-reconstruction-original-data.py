Author: Elizabeth N.

import nibabel as nib
import numpy as np
import os

# Path to the main directory containing subject folders
input_dir = '/Users/elizabethnemeti/Desktop/Reconstructed-3D-original'
output_dir = '/Users/elizabethnemeti/Desktop/Reconstructed-4D-original'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over each subject folder
for subject in os.listdir(input_dir):
    subject_folder = os.path.join(input_dir, subject)
    
    if os.path.isdir(subject_folder):  # Ensure it's a directory
        print(f"Processing subject: {subject}")
        
        # Get all 3D NIfTI files for the subject, sorted by time point
        time_point_files = sorted([f for f in os.listdir(subject_folder) if f.endswith('.nii.gz')])
        
        # Load the 3D volumes and stack them into a 4D array
        volumes = []
        for file in time_point_files:
            filepath = os.path.join(subject_folder, file)
            nii = nib.load(filepath)
            volumes.append(nii.get_fdata())  # Get data as numpy array
        
        # Combine into a single 4D volume (x, y, z, time)
        volume_4d = np.stack(volumes, axis=-1)
        
        # Save the 4D volume
        affine = nib.load(os.path.join(subject_folder, time_point_files[0])).affine  # Use affine from the first file
        spacing = nib.load(os.path.join(subject_folder, time_point_files[0])).header.get_zooms()
        time_spacing = 1.0  # Define temporal spacing
        
        # Create a NIfTI object and set the metadata
        nii_4d = nib.Nifti1Image(volume_4d, affine)
        nii_4d.header.set_zooms(spacing[:3] + (time_spacing,))  # Add temporal spacing
        
        # Save to the subject's output folder
        subject_output_folder = os.path.join(output_dir, subject)
        os.makedirs(subject_output_folder, exist_ok=True)
        output_file = os.path.join(subject_output_folder, f"{subject}_4D_reconstructed.nii.gz")
        nib.save(nii_4d, output_file)
        
        print(f"4D volume saved to {output_file}")
