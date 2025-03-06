from PIL import Image
import os

# Define directories
input_dir = r"E:\PythonProjects\gsoc-2024\data\Synthetic-Data-600"  # Directory containing input images/masks
output_dir = r"E:\PythonProjects\gsoc-2024\data\Synthetic-Data-600"  # Directory to save converted files
os.makedirs(output_dir, exist_ok=True)

# Process all files in the input directory
for file_name in os.listdir(input_dir):
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name)

    try:
        with Image.open(input_path) as img:
            # Convert to grayscale (1 channel)
            gray_img = img.convert('L')
            # Save the processed image
            gray_img.save(output_path)
            print(f"Converted and saved: {output_path}")
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

for file_name in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file_name)
    with Image.open(file_path) as img:
        print(f"{file_name}: {img.size}, {img.mode}")

