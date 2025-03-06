import os
import cv2
import matplotlib.pyplot as plt

# Define the path to the directory containing the images
image_directory = '../data/training_LOO'

# Define the path for the new directory to save histograms
histogram_directory = '../data/histograms'

# Create the directory if it doesn't exist
os.makedirs(histogram_directory, exist_ok=True)

# Loop through the files in the directory
for filename in os.listdir(image_directory):
    if "_image.png" in filename:  # Process only image files
        image_path = os.path.join(image_directory, filename)
        
        # Load the image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Assuming the images are grayscale
        
        # Calculate the histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        
        # Plot the histogram with filled style
        plt.figure()
        plt.title(f"Histogram for {filename}")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.fill_between(range(256), hist.flatten(), color='blue')
        plt.xlim([0, 256])
        
        # Save the histogram plot
        histogram_path = os.path.join(histogram_directory, f"{filename}_histogram.png")
        plt.savefig(histogram_path)
        
        # Optionally show the plot
        # plt.show()
        
        # Close the plot to free up memory
        plt.close()

print(f"Histograms saved in {histogram_directory}")
