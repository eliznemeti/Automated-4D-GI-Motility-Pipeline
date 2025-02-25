import os
import shutil

# Path to the main directory containing subject folders
source_dir = '/Users/elizabethnemeti/Desktop/roberta-per-subject-data-2d'
output_dir = '/Users/elizabethnemeti/Desktop/LOOCV-Splits'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all subject folders
subjects = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])

# Leave-One-Out Cross-Validation
for leave_out_subject in subjects:
    print(f"Processing LOOCV iteration with validation subject: {leave_out_subject}")
    
    # Create LOO-specific folder with training_LOO and validation_LOO subfolders
    loo_dir = os.path.join(output_dir, f"LOO_{leave_out_subject}")
    train_dir = os.path.join(loo_dir, 'training_LOO')
    val_dir = os.path.join(loo_dir, 'validation_LOO')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Loop through subjects
    for subject in subjects:
        source_path = os.path.join(source_dir, subject)
        if subject == leave_out_subject:
            # Copy all files directly to validation_LOO
            for filename in os.listdir(source_path):
                src_file = os.path.join(source_path, filename)
                dst_file = os.path.join(val_dir, filename)
                shutil.copy2(src_file, dst_file)
            print(f"Copied {subject} data to validation_LOO.")
        else:
            # Copy all files directly to training_LOO
            for filename in os.listdir(source_path):
                src_file = os.path.join(source_path, filename)
                dst_file = os.path.join(train_dir, filename)
                shutil.copy2(src_file, dst_file)
            print(f"Copied {subject} data to training_LOO.")

print("Leave-One-Out Cross-Validation data split completed!")
