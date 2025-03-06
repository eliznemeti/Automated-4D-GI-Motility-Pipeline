import os
import subprocess
import glob

# This script will loop and generate synthetic data for the entire dataset
# Specify desired number of samples with n_samples
# Run with no additional parameters using: python generate_synthetic.py

def synthetic_data_generation(n_samples=50):
    
    # Path to the dataset (change this if necessary)
    images = glob.glob(r"C:\Users\User\Desktop\singan-seg\Input\data-RGBA\*.png")  # Adjust this path as needed
    
    if not images:
        print("No images found. Please check the directory and image file paths.")
        return
    
    for img in images:
        img_name = os.path.basename(img)  # Get the image name

        print(f"Generating synthetic data for: {img_name}")
        
        # Command to train model and generate synthetic data for each image
        python_command = f"python main_train.py --input_name {img_name} --nc_z 4 --nc_im 4 --gpu_id 0"
        
        try:
            print(f"Running command: {python_command}")
            result = subprocess.run(python_command, shell=True, check=True)
            print(f"Command completed successfully for {img_name}\n")
        
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing {img_name}: {e}")
            continue  # Continue processing the next image if an error occurs

# Call the function to start the loop
synthetic_data_generation(n_samples=50)
