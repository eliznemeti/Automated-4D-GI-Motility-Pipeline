from PIL import Image, ImageEnhance
import os

def enhance_contrast(input_dir, output_dir, factor):
    """
    Enhance the contrast of all images in the input directory and save them to the output directory.
    
    :param input_dir: Directory containing original images.
    :param output_dir: Directory to save the enhanced images.
    :param factor: Contrast enhancement factor (1.0 means no change, >1.0 increases contrast).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust to your image file types
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path)
            enhancer = ImageEnhance.Contrast(image)
            enhanced_image = enhancer.enhance(factor)
            enhanced_image.save(os.path.join(output_dir, filename))
            print(f"Enhanced {filename} with a factor of {factor}")

# Usage
input_directory = '../data/brightened-images-roberta'  # Your original images directory
output_directory = '../data/enhanced-images-roberta'  # Directory to save enhanced images
contrast_factor = 1.5  # Adjust this factor to control the level of enhancement

enhance_contrast(input_directory, output_directory, contrast_factor)
