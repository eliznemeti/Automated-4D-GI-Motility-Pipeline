# Updated dataloader script: dataloaders.py

import torch
import cv2
import os
import glob
import numpy as np
from typing import Any
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from src.utils.data_utils import read_image, resize_torch_tensor  

class MadisonDataset(Dataset):

    def __init__(self, image_paths) -> None:
        
        self.image_paths = image_paths
        self.filtered_paths = []

        img_dict = {}
        for img in self.image_paths:

            re = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            if len(re.shape) == 2 :
                self.filtered_paths.append(img)

    def __len__(self) -> int:
        return len(self.filtered_paths)

    def __getitem__(self, index) -> Any:

        #img = cv2.imread(self.image_paths[index], cv2.IMREAD_UNCHANGED)
        img = read_image(self.filtered_paths[index])
        img = resize_torch_tensor(img, 256, 256)

        return img

class MadisonDatasetLabeled(Dataset):
    def __init__(self, segmentation_path, augment=False) -> None:
        self.image_paths = sorted(glob.glob(os.path.join(segmentation_path, '*image*.png')))
        self.mask_paths = sorted(glob.glob(os.path.join(segmentation_path, '*mask*.png')))
        
        assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks do not match."
        
        self.target_size = (256, 256)
        
        # Define transformations
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_size, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_size, interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float()),  # Binarize the mask
        ])
        
        self.augment = augment
        if self.augment:
            self.augmentation_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index) -> Any:
        # Load image and mask using cv2
        img = cv2.imread(self.image_paths[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {self.image_paths[index]}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found at path: {self.mask_paths[index]}")
        
        # Convert to float32 and scale to [0, 1]
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        # Apply transformations
        img = self.image_transform(img)
        mask = self.mask_transform(mask)
        
        if self.augment:
            # Apply the same random transformation to both image and mask
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            img = self.augmentation_transforms(img)
            
            torch.manual_seed(seed)
            mask = self.augmentation_transforms(mask)
        
        return img, mask, self.image_paths[index]

if __name__ == '__main__':
    # Example usage for testing the data loader
    segmentation_path = r'E:\\PythonProjects\\gsoc-2024\\data\\images-roberta'
    dataset = MadisonDatasetLabeled(segmentation_path=segmentation_path, augment=False)
    dataloader = DataLoader(dataset=dataset, batch_size=5, shuffle=True)
    data = next(iter(dataloader))

    imgs, masks, paths = data
    print(f"Number of batches: {len(dataloader)}")
    print(f"Image batch shape: {imgs.shape}")  # Expected: [5, 1, 256, 256]
    print(f"Mask batch shape: {masks.shape}")  # Expected: [5, 1, 256, 256]
