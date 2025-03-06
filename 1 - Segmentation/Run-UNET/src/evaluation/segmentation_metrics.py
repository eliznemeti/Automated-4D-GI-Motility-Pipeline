import numpy as np
import torch

def dice_coefficient(pred, target, threshold=0.9, epsilon=1e-6):
    """
    Computes the Dice Coefficient between predicted and target masks.
    Args:
        pred (torch.Tensor): Model outputs (logits or probabilities).
        target (torch.Tensor): Ground truth binary masks.
        threshold (float, optional): Threshold to binarize predictions. Defaults to 0.5.
        epsilon (float, optional): Small value to avoid division by zero. Defaults to 1e-6.
    Returns:
        float: Mean Dice Coefficient across the batch.
    """
    # Apply sigmoid activation if necessary
    pred = torch.sigmoid(pred) if pred.max() > 1 else pred

    # Binarize predictions and ground truth
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    # Flatten tensors
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    # Compute intersection and union for each sample
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    # Compute Dice score per sample and average
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()
