import numpy as np
from scipy.io import savemat

# Path to the .npy file
npy_path = r"E:\PythonProjects\gsoc-2024\reports\FULL_2D_SEG_LOO_32\predictions\FD_032_reconstructed.npy"

# Load the .npy file
volume = np.load(npy_path)
print(f"Volume shape: {volume.shape}")  # Confirm the shape

# Path for the .mat file
mat_path = r"E:\PythonProjects\gsoc-2024\reports\FULL_2D_SEG_LOO_32\predictions\FD_032_reconstructed.mat"

# Save as .mat
savemat(mat_path, {'reconstructed_volume': volume})  # 'reconstructed_volume' is the variable name in the .mat file
print(f"Saved .mat file to {mat_path}")
