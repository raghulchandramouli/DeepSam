import os
import torch
import random
from torch.utils.data import Dataset
from PIL import Image, Image as PILImage
from torchvision import transforms
import numpy as np

def center_crop_full(img, crop_size, output_size, is_mask=False):
    w, h = img.size
    cw, ch = crop_size
    left = (w - cw) // 2
    top = (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))
    resample = PILImage.Resampling.NEAREST if is_mask else PILImage.Resampling.BILINEAR
    return cropped.resize(output_size, resample)

def random_crop_full(img, crop_size, output_size, is_mask=False):
    w, h = img.size
    cw, ch = crop_size
    if w < cw or h < ch:
        return center_crop_full(img, (min(w, cw), min(h, ch)), output_size, is_mask)
    left = random.randint(0, w - cw)
    top = random.randint(0, h - ch)
    cropped = img.crop((left, top, left + cw, top + ch))
    resample = PILImage.Resampling.NEAREST if is_mask else PILImage.Resampling.BILINEAR
    return cropped.resize(output_size, resample)

class AdvancedAugmentation:
    def __init__(self, crop_size=(400, 400), output_size=(512, 512), p_center=0.5):
        self.crop_size = crop_size
        self.output_size = output_size
        self.p_center = p_center

    def __call__(self, img, mask=None):
        if random.random() < self.p_center:
            aug_img = center_crop_full(img, self.crop_size, self.output_size, is_mask=False)
            aug_mask = center_crop_full(mask, self.crop_size, self.output_size, is_mask=True) if mask else None
        else:
            aug_img = random_crop_full(img, self.crop_size, self.output_size, is_mask=False)
            aug_mask = random_crop_full(mask, self.crop_size, self.output_size, is_mask=True) if mask else None
        return aug_img, aug_mask

class InpaintingDetectionDataset(Dataset):
    def __init__(self, data_dir, target_size=(512, 512), use_augmentation=True):
        self.data_dir = data_dir
        self.mask_dir = os.path.join(data_dir, "masks")
        self.inpaint_dir = os.path.join(data_dir, "inpainting")
        
        # Check if directories exist
        if not os.path.exists(self.mask_dir):
            raise ValueError(f"Masks directory not found: {self.mask_dir}")
        if not os.path.exists(self.inpaint_dir):
            raise ValueError(f"Inpainting directory not found: {self.inpaint_dir}")
        
        # Get files from inpainting directory (assuming it has the images)
        self.files = sorted([f for f in os.listdir(self.inpaint_dir) if f.endswith('.jpg')])
        self.target_size = target_size
        
        print(f"Found {len(self.files)} files in dataset")
        
        # Augmentation
        self.augmentation = AdvancedAugmentation(
            crop_size=(target_size[0]//2, target_size[1]//2),
            output_size=target_size,
            p_center=0.5
        ) if use_augmentation else None
        
        # Transforms
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        base_name = filename.split('.')[0]
        
        # Load images - use inpainted as both real and inpainted for now
        inpaint_img = Image.open(os.path.join(self.inpaint_dir, filename)).convert('RGB')
        mask_img = Image.open(os.path.join(self.mask_dir, f"{base_name}.png")).convert('L')
        
        # For real image, we'll use the inpainted image (you can modify this logic)
        real_img = inpaint_img.copy()
        
        # Apply augmentation
        if self.augmentation:
            real_img, mask_img = self.augmentation(real_img, mask_img)
            inpaint_img, _ = self.augmentation(inpaint_img, mask_img)
        else:
            real_img = real_img.resize(self.target_size)
            inpaint_img = inpaint_img.resize(self.target_size)
            mask_img = mask_img.resize(self.target_size)
        
        # Transform to tensors
        real_tensor = self.normalize(real_img)
        inpaint_tensor = self.normalize(inpaint_img)
        mask_tensor = self.mask_transform(mask_img)
        
        # Binary mask
        mask_binary = (mask_tensor > 0.5).float()
        
        return {
            'real': real_tensor,
            'inpainted': inpaint_tensor,
            'mask': mask_binary,
            'has_inpainting': mask_binary.sum() > 0
        }
