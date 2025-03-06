import os
import shutil

# Path to the root folder containing the subfolders with brightness and contrast
root_dir = r'E:\PythonProjects\gsoc-2024\reports\seg\gridsearchresults'

# Path to the destination folder where we will copy and rename the dice coefficient images
dest_dir = r'E:\PythonProjects\gsoc-2024\reports\seg\gridsearchresults\dice_coefs'

# Create the destination folder if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Walk through all subdirectories in the root folder
for dirpath, dirnames, filenames in os.walk(root_dir):
    print(f"Checking folder: {dirpath}")  # Log current directory being checked
    
    # Skip the destination directory itself to avoid copying into it
    if dirpath == dest_dir:
        continue
    
    # Check if any JPG files exist that include "dice" in the filename
    for filename in filenames:
        if filename.lower().endswith('.jpg') and "dice" in filename.lower():  # Look for JPG files with "dice" in the name
            image_path = os.path.join(dirpath, filename)
            
            # Use the folder name (relative path) as the new image name
            folder_name = os.path.basename(dirpath)
            new_image_name = f'{folder_name}.jpg'
            
            # Full path for the destination
            dest_image_path = os.path.join(dest_dir, new_image_name)
            
            # Copy and rename the image
            shutil.copyfile(image_path, dest_image_path)
            print(f"Copied from {image_path} to {dest_image_path}")
