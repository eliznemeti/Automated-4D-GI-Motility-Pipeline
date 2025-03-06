import matplotlib.pyplot as plt
from itertools import product
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import glob
import torch
import os
import re

print("viz_utils.py is being imported successfully.")

def plot_results(imgs, recons, save_path, epoch, batch):

    bs = 8 #imgs.size(0)
    fig, axes = plt.subplots(nrows=2,ncols=bs,figsize=(bs,15))

    for i, (row,col) in enumerate(product(range(2),range(bs))):

        if row == 0:
            axes[row][col].imshow(np.transpose(imgs[col].detach().cpu().numpy(),(1,2,0)))
            if col == 0:
                axes[row][col].set_ylabel('Original Image',fontsize=15,fontweight='bold')
        
        elif row == 1:
            axes[row][col].imshow(np.transpose(recons[col].detach().cpu().numpy(),(1,2,0)))

            if col == 0:
                axes[row][col].set_ylabel('Reconstructed Image',fontsize=15,fontweight='bold')

            
        axes[row][col].set_yticks([])
        axes[row][col].set_xticks([])

    plt.subplots_adjust(wspace=0,hspace=0)
    plt.savefig(os.path.join(save_path,f'fig_{epoch}_{batch}.jpg'),format='jpg',bbox_inches='tight', pad_inches=0, dpi=100)
    plt.show() 
    plt.close()

def visualize_samples(tensor, save_path, epoch, batch):

    tensor = tensor.cpu().numpy()
    tensor = tensor.squeeze(1)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(5):
        axes[i].imshow(tensor[i])
        axes[i].axis('off')
    print("SaveDIR:",os.path.join(save_path,f'fig_{epoch}_{batch}.jpg'))
    plt.savefig(os.path.join(save_path,f'fig_{epoch}_{batch}.jpg'),format='jpg',bbox_inches='tight', pad_inches=0, dpi=100)
    plt.show() 
    plt.close()

def save_tensor_as_jpg(tensor, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    length_dir = glob.glob(os.path.join(save_dir,'*'))

    for idx, img in enumerate(tensor):

        transform = transforms.ToPILImage()
        image = transform(img)
        image.save(os.path.join(save_dir,f'{idx+len(length_dir)}.jpeg'), 'JPEG')


def plot_from_dir(folder_path, num_sample=25):

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png') or f.lower().endswith('.jpeg')]
    
    if len(image_files) > num_sample:
        image_files = random.sample(image_files, num_sample)
    

    grid_size = int(num_sample ** 0.5)
    if grid_size ** 2 < num_sample:
        grid_size += 1

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    for ax, img_file in zip(axes, image_files):
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path)
        
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    for ax in axes[len(image_files):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('figure-gray.jpg',format='jpg',bbox_inches='tight', pad_inches=0, dpi=100)
    plt.show()

def visualize_tensor(tensor):
    
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    tensor = tensor.numpy()
    tensor = tensor.squeeze(1)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(5):
        axes[i].imshow(tensor[i], cmap='gray')
        axes[i].axis('off')
    plt.show()

    
def visualize_predictions(images, masks, outputs, save_path, epoch, batch_idx, paths):
    # Convert tensors to CPU and numpy arrays
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    outputs = torch.sigmoid(outputs).cpu().numpy() > 0.5  # Apply threshold to outputs

    batch_size = images.shape[0]
    fig, axs = plt.subplots(3, batch_size, figsize=(batch_size * 3, 9))

    for i in range(batch_size):
        # Extract the time point using a regular expression
        image_filename = os.path.basename(paths[i])

        # Regular expression to find the number after 'time_'
        match = re.search(r'_time_(\d+)', image_filename)
        if match:
            time_point = match.group(1)  # Extract the number after 'time_'
        else:
            time_point = 'Unknown'  # Fallback if the regex does not match

        # Display original image
        axs[0, i].imshow(images[i][0], cmap='gray')  # Assuming grayscale images
        axs[0, i].axis('off')
        axs[0, i].set_title(f"Time {time_point}")

        # Display ground truth mask
        axs[1, i].imshow(masks[i][0], cmap='gray')
        axs[1, i].axis('off')
        axs[1, i].set_title("Ground Truth")

        # Display predicted mask
        axs[2, i].imshow(outputs[i][0], cmap='gray')
        axs[2, i].axis('off')
        axs[2, i].set_title("Segmentation Output")

    plt.suptitle(f'Epoch {epoch} Predictions')
    plt.tight_layout()  # Adjust layout to make sure titles and images do not overlap
    save_filename = os.path.join(save_path, f'fig_{epoch}_{batch_idx}.jpg')
    plt.savefig(save_filename, format='jpg', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)  # Close the figure to free memory
    print(f"Saved visualization to {save_filename}")