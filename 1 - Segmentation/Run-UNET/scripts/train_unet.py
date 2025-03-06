import numpy as np
import torch
import os
from tqdm import tqdm
import sys
import cv2
import csv
sys.path.append(os.path.dirname(os.getcwd()))

import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from src.dataloader.dataloaders import MadisonDatasetLabeled
from src.models.unet import BaseUNet
from src.utils.viz_utils import visualize_predictions
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
    plot_path = os.path.join(plot_dir, args.exp_id, f'{metric}_curve.jpg')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, format='jpg', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

    print(f'{metric.capitalize()} plot saved to {plot_path}')

def parse_path(path):
    filename = os.path.basename(path)  # Get the filename, e.g., 'FD_027_time_1_slice_7_image.png'
    parts = filename.split('_')  # Split by underscores
    subject = parts[0]  # e.g., 'FD_027'
    time_point = parts[2]  # e.g., '1'
    slice_idx = parts[4]  # e.g., '7'
    return subject, time_point, slice_idx


def train_model(model, train_loader, val_loader, optimizer, criterion, device, args):
    exp_dir = os.path.join(PLOT_DIRECTORY, args.exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'model'), exist_ok=True)

    train_loss_history = []
    val_loss_history = []
    dice_coef_history = []
    best_dice_predictions = []  # Store predictions for the best Dice epoch

    best_dice = -float('inf')  # Initialize the best Dice coefficient to a very low value
    patience_counter = 0
    patience_limit = 10
    best_dice_epoch = -1  # Track the best Dice epoch

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

        train_loss = train_loss / len(train_loader.dataset)
        train_loss_history.append(train_loss)

        model.eval()
        val_loss = 0.0
        dice_coefficients = []
        current_epoch_predictions = []  # Store predictions for the current epoch
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images, masks, paths = batch
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                # Loss calculation
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                # Save predictions to memory (not disk yet)
                for i, path in enumerate(paths):
                    predicted_mask = (torch.sigmoid(outputs[i]) > 0.5).cpu().numpy()
                    predicted_mask = (predicted_mask.squeeze() * 255).astype(np.uint8)

                    # Store predictions
                    current_epoch_predictions.append((predicted_mask, path))

                # Compute Dice coefficient per sample
                for i in range(images.size(0)):
                    dice_score = dice_coefficient(outputs[i], masks[i], threshold=0.1)
                    dice_coefficients.append(dice_score)

        # Average Dice across all validation samples
        dice_mean = np.mean(dice_coefficients)
        dice_coef_history.append(dice_mean)
        val_loss = val_loss / len(val_loader.dataset)
        val_loss_history.append(val_loss)

        print(f"Dice Coefficient for Epoch {epoch}: {dice_mean:.4f}")
        print(f'Epoch {epoch+1}/{args.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save the model and predictions only if Dice coefficient improves
        if dice_mean > best_dice:
            best_dice = dice_mean
            best_dice_epoch = epoch
            best_dice_predictions = current_epoch_predictions  # Update best predictions

            save_path = os.path.join(exp_dir, 'model', 'best_model.pt')
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch + 1} with Dice coefficient {best_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"Stopping early after {epoch + 1} epochs due to no improvement in Dice coefficient.")
            break

    # Save predictions for the best Dice epoch to disk
    save_dir = os.path.join(exp_dir, 'predictions')
    os.makedirs(save_dir, exist_ok=True)
    for predicted_mask, original_path in best_dice_predictions:
        # Save the predicted mask with a consistent filename
        original_filename = os.path.basename(original_path).replace("_image.png", "_predicted.png")
        save_path = os.path.join(save_dir, original_filename)
        cv2.imwrite(save_path, predicted_mask)

    print(f"Predictions for the best Dice epoch ({best_dice_epoch + 1}) saved to {save_dir}")

    # Save best Dice epoch info to a text file
    best_dice_path = os.path.join(exp_dir, 'best_dice_epoch.txt')
    with open(best_dice_path, 'w') as f:
        f.write(f"Best Epoch: {best_dice_epoch + 1}\n")
        f.write(f"Dice Coefficient: {best_dice:.4f}\n")
    print(f"Best Dice epoch details saved to {best_dice_path}")

    # Plot and save Dice coefficient curve
    plot_metric(x=dice_coef_history,
                label="Dice Coefficient",
                plot_dir=PLOT_DIRECTORY,
                args=args,
                metric='dice_coeff')

    print("Training completed.")

if __name__ == "__main__":
    args = train_arg_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the datasets
    print(f"Loading training data from: {TRAINING_LOO}")
    train_dataset = MadisonDatasetLabeled(TRAINING_LOO, augment=True)
    print(f"Loading validation data from: {VALIDATION_LOO}")
    val_dataset = MadisonDatasetLabeled(VALIDATION_LOO, augment=False)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

    model = BaseUNet(in_channels=1, out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    train_model(model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                args=args)
