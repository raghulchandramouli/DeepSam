import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class InpaintingDataset(Dataset):
    
    def __init__(
        self, 
        data_root = "/mnt/g/Authenta/data/authenta-inpainting-detection/single_mask",
        transform = None
    ):
        
        self.image_dir = os.path.join(data_root, "Inpainting")
        self.mask_dir = os.path.join(data_root, "masks")
        
        # print(f"Image dir: {self.image_dir}")
        # print(f"Mask dir: {self.mask_dir}")
        # print(f"Image files: {os.listdir(self.image_dir)}")
        # print(f"Mask files: {os.listdir(self.mask_dir)}")
        
        self.image_files = {
            os.path.splitext(f)[0].split('_')[0]: f
            for f in os.listdir(self.image_dir) if f.endswith('.jpg')
        }
        
        self.mask_files = {
            os.path.splitext(f)[0].split('_')[0]: f
            for f in os.listdir(self.mask_dir) if f.endswith('.png')
        } 

        
        common_ids = set(self.image_files.keys()) & set(self.mask_files.keys())
        self.matching_pairs = sorted([
            (self.image_files[i], self.mask_files[i]) for i in common_ids
        ])
        
        self.target_size = (1024, 1024)
        self.transform = transform
        print(f"Found {len(self.matching_pairs)} matching image-mask pairs.")
        
        
    def __len__(self):
        return len(self.matching_pairs)
    
    def __getitem__(self, idx):
        image_filename, mask_filename = self.matching_pairs[idx]
        
        image_path = os.path.join(self.image_dir, image_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        image = Image.open(image_path).convert("RGB").resize(self.target_size)
        mask = Image.open(mask_path).convert("L").resize(self.target_size)
        
        image = np.array(image)/ 255.0
        mask = np.array(mask) > 128
        
        ys, xs = np.where(mask)
        if len(xs) == 0:
            point = [random.randint(0, self.target_size[0] - 1),
                     random.randint(0, self.target_size[1] - 1)]
            
        else:
            i = random.randint(0, len(xs) - 1)
            point = [xs[i], ys[i]]
            
        return {
            "image" : torch.tensor(image).permute(2, 0, 1).float(),
            "mask"  : torch.tensor(mask).float(),
            "point" : torch.tensor(point).float(),
        }

