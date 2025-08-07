import os
import warnings
import random
from PIL import Image, Image as PILImage
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

# --- Augmentation Functions ---
def center_crop_full(img, crop_size, output_size, is_mask=False):
    """
    Center-crop `crop_size` then resize back to `output_size`.
    Uses NEAREST for masks, BILINEAR for images.
    """
    w, h = img.size
    cw, ch = crop_size
    left = (w - cw) // 2
    top  = (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))
    resample = PILImage.Resampling.NEAREST if is_mask else PILImage.Resampling.BILINEAR
    return cropped.resize(output_size, resample)


def random_crop_full(img, crop_size, output_size, is_mask=False):
    """
    Random-crop `crop_size` then resize back to `output_size`.
    """
    w, h = img.size
    cw, ch = crop_size
    if w < cw or h < ch:
        return center_crop_full(img, (min(w, cw), min(h, ch)), output_size, is_mask)
    left = random.randint(0, w - cw)
    top  = random.randint(0, h - ch)
    cropped = img.crop((left, top, left + cw, top + ch))
    resample = PILImage.Resampling.NEAREST if is_mask else PILImage.Resampling.BILINEAR
    return cropped.resize(output_size, resample)


class AdvancedAugmentation:
    """
    Randomly apply either center crop or random crop to image+mask pairs.

    Args:
        crop_size (tuple): Size of the crop (width, height).
        output_size (tuple): Final resize dimensions (width, height).
        p_center (float): Probability of choosing center crop.
    """
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


class InpaintingDataset(Dataset):
    def __init__(
        self,
        image_dir=None,
        mask_dir=None,
        image_files=None,
        mask_files=None,
        transform=None,
        target_size=(1024, 1024),
        padding=True,
        validate_files=True,
        skip_missing=True
    ):
        self.target_size = target_size
        self.padding = padding
        self.validate_files = validate_files
        self.skip_missing = skip_missing

        # Determine file lists
        if image_files and mask_files:
            assert len(image_files) == len(mask_files), "Image and mask file counts do not match."
            self.image_files = image_files
            self.mask_files = mask_files
            if self.validate_files:
                self._validate_file_lists()

        elif image_dir:
            self._setup_from_directory(image_dir, mask_dir)
        else:
            raise ValueError("Either image_files/mask_files or image_dir must be specified.")

        if len(self.image_files) == 0:
            raise ValueError("No valid image files found after validation.")

        # Use custom transform or default advanced augmentation
        self.augment = transform or AdvancedAugmentation(
            crop_size=(target_size[0]//2, target_size[1]//2),
            output_size=target_size,
            p_center=0.5
        )

        print(f"Dataset initialized with {len(self.image_files)} valid samples.")

    def _validate_file_lists(self):
        valid_images, valid_masks = [], []
        for idx, (img_path, msk_path) in enumerate(zip(self.image_files, self.mask_files)):
            img_ok = os.path.isfile(img_path)
            msk_ok = os.path.isfile(msk_path) if msk_path else True
            if img_ok and msk_ok:
                valid_images.append(img_path)
                valid_masks.append(msk_path)
            else:
                if self.skip_missing:
                    warnings.warn(f"Skipping sample {idx}: Missing {img_path if not img_ok else ''} {msk_path if not msk_ok else ''}")
                else:
                    raise FileNotFoundError(f"Missing files for sample {idx}")
        self.image_files = valid_images
        self.mask_files = valid_masks if any(valid_masks) else None

    def _setup_from_directory(self, image_dir, mask_dir):
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory not found: {image_dir}")
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        all_imgs = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(exts)])
        if not all_imgs:
            raise ValueError(f"No image files found in {image_dir}")
        print(f"Found {len(all_imgs)} images in {image_dir}")
        # Pair masks
        if mask_dir and os.path.exists(mask_dir):
            mask_map = {''.join(filter(str.isdigit, f)): os.path.join(mask_dir, f)
                        for f in os.listdir(mask_dir) if f.lower().endswith(exts)}
            imgs, msks = [], []
            for img_path in all_imgs:
                key = ''.join(filter(str.isdigit, os.path.basename(img_path)))
                if key in mask_map:
                    imgs.append(img_path)
                    msks.append(mask_map[key])
                elif not self.skip_missing:
                    raise ValueError(f"No mask for image {img_path}")
            self.image_files = imgs
            self.mask_files = msks
            print(f"Paired {len(imgs)} image-mask samples.")
        else:
            self.image_files = all_imgs
            self.mask_files = None
            print("No mask directory; proceeding without masks.")

    def __len__(self):
        return len(self.image_files)

    def pad_image(self, tensor_img):
        h, w = tensor_img.shape[-2:]
        th, tw = self.target_size
        pad_h, pad_w = max(th-h,0), max(tw-w,0)
        padding = [pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2]
        return F.pad(tensor_img, padding)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        mask = None
        if self.mask_files:
            mask = Image.open(self.mask_files[idx]).convert('L')
        # Augment
        image_aug, mask_aug = self.augment(image, mask)
        # To tensor
        image_t = transforms.ToTensor()(image_aug)
        if self.padding:
            image_t = self.pad_image(image_t)
        result = {'image': image_t}
        if mask_aug:
            mask_t = transforms.ToTensor()(mask_aug)
            if self.padding:
                mask_t = self.pad_image(mask_t)
            result['mask'] = mask_t
        # Add point for SAM
        result['point'] = torch.tensor([self.target_size[1]//2, self.target_size[0]//2]).float()
        result['image_path'] = img_path
        if self.mask_files:
            result['mask_path'] = self.mask_files[idx]
        return result

    def get_statistics(self):
        return {
            'total_samples': len(self),
            'has_masks': bool(self.mask_files),
            'target_size': self.target_size,
            'padding': self.padding
        }
