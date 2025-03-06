import numpy as np
import torch
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from src.dataloader.dataloaders import MadisonDatasetLabeled
from src.models.unet import BaseUNet
from src.utils.viz_utils import plot_results, visualize_predictions
from src.utils.args_utils import train_arg_parser
from src.evaluation.segmentation_metrics import dice_coefficient
from src.utils.variable_utils import PLOT_DIRECTORY, TRAINING_LOO, VALIDATION_LOO

# Define the plot_metric function
def plot_metric(x, label, plot_dir, args, metric):
    plt.figure()
    plt.plot(x, label=label)
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} Over Epochs')
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(plot_dir, f'{metric}_curve.jpg')  # Ensure no redundant paths
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    plt.savefig(plot_path, format='jpg', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

    print(f'{metric.capitalize()} plot saved to {plot_path}')

def grid_search_brightness_contrast(existing_dir, brightness_range, contrast_range, train_func):
    best_performance = 0
    best_params = None
    performance_log = {}
    
    for brightness in brightness_range:
        for contrast in contrast_range:
            adjusted_output_dir = os.path.join(existing_dir, f'bright_{brightness}_contrast_{contrast}')
            
            # Directly use the existing dataset directories
            performance = train_func(adjusted_output_dir, brightness, contrast)
            
            # Log the performance for this combination
            performance_log[(brightness, contrast)] = performance
            
            if performance > best_performance:
                best_performance = performance
                best_params = {'brightness': brightness, 'contrast': contrast}
    
    # Print out the results for each combination
    print("\nGrid Search Results:")
    for (brightness, contrast), score in performance_log.items():
        print(f"Brightness {brightness}, Contrast {contrast}: Dice Score = {score:.4f}")
    
    # Print the best combination
    print(f"\nBest performance {best_performance:.4f} with brightness {best_params['brightness']} and contrast {best_params['contrast']}")

def train_model(model, train_loader, val_loader, optimizer, criterion, device, args, brightness, contrast):
    base_dir = os.path.join(PLOT_DIRECTORY, 'seg', 'gridsearchresults')
    exp_dir = os.path.join(base_dir, f'bright_{brightness}_contrast_{contrast}')
    print(f"Experiment directory: {exp_dir}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'model'), exist_ok=True)

    train_loss_history = []
    val_loss_history = []
    dice_coef_history = []
    val_images = []

    for epoch in range(args.epoch):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}")):
            images, masks, paths = batch
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        
        save_path = os.path.join(exp_dir, 'model', f'model_{epoch}.pt')
        torch.save(model.state_dict(), save_path)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_loss_history.append(train_loss)

        model.eval()
        val_loss = 0.0
        dice_coefficients = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images, masks, paths = batch
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                visualize_predictions(images, masks, outputs, exp_dir, epoch, batch_idx, paths)
                dice_score = dice_coefficient(masks, outputs)
                dice_coefficients += dice_score
        
        dice_coefficients = dice_coefficients / len(val_loader.dataset)
        dice_coef_history.append(dice_coefficients)
        val_loss = val_loss / len(val_loader.dataset)
        val_loss_history.append(val_loss)

        print(f"Dice Coefficient for Epoch {epoch}: {dice_coefficients:.4f}")
        print(f'Epoch {epoch+1}/{args.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    plt.figure()
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, f'train_loss_curves_brightness_{brightness}_contrast_{contrast}.jpg'))
    plt.close()

    plot_metric(x=dice_coef_history, 
                label="Dice Coeff",
                plot_dir=exp_dir,
                args=args,
                metric='dice_coeff')

    with open(os.path.join(exp_dir, 'all_image_paths.txt'), 'w') as f:
        for path in val_images:
            f.writelines(path + "\n")

    print("Training completed.")
    return np.mean(dice_coef_history)

def train_func(adjusted_output_dir, brightness, contrast):
    training_dir = os.path.join(adjusted_output_dir, 'training_LOO')
    validation_dir = os.path.join(adjusted_output_dir, 'validation_LOO')
    
    train_dataset = MadisonDatasetLabeled(training_dir, augment=True)
    val_dataset = MadisonDatasetLabeled(validation_dir, augment=False)

    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

    if len(val_loader.dataset) == 0:
        print(f"No validation data found in {validation_dir}!")
        raise ValueError("Validation dataset is empty.")
    
    return train_model(model, train_loader, val_loader, optimizer, criterion, device, args, brightness, contrast)

if __name__ == "__main__":
    args = train_arg_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaseUNet(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    brightness_range = np.arange(5.5, 6.0, 0.5)
    contrast_range = np.arange(1.0, 6.0, 0.5)

    grid_search_brightness_contrast('../data/CB-datasets', 
                                    brightness_range, 
                                    contrast_range, 
                                    train_func)