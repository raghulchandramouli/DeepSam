import random
from PIL import Image
import matplotlib.pyplot as plt

# --- Augmentation Functions ---
from PIL import Image as PILImage

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
        # Decide which augmentation to apply
        if random.random() < self.p_center:
            aug_img = center_crop_full(img, self.crop_size, self.output_size, is_mask=False)
            aug_mask = center_crop_full(mask, self.crop_size, self.output_size, is_mask=True) if mask else None
        else:
            aug_img = random_crop_full(img, self.crop_size, self.output_size, is_mask=False)
            aug_mask = random_crop_full(mask, self.crop_size, self.output_size, is_mask=True) if mask else None
        return aug_img, aug_mask


if __name__ == "__main__":
    # Demo: load one image-mask pair and show 4 randomized outputs
    img = Image.open("/mnt/g/Authenta/data/authenta-inpainting-detection/patch_crop/image_tile/000000000400.png").convert("RGB")
    mask = Image.open("/mnt/g/Authenta/data/authenta-inpainting-detection/patch_crop/mask_tile/000000000400.png").convert("L")


    aug = AdvancedAugmentation(crop_size=(256, 256), output_size=(512, 512), p_center=0.5)

    # Generate 4 different augmentations
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(4):
        out_img, out_mask = aug(img, mask)
        axes[0, i].imshow(out_img)
        axes[0, i].set_title(f"Aug Img #{i+1}")
        axes[0, i].axis('off')

        axes[1, i].imshow(out_mask, cmap='gray')
        axes[1, i].set_title(f"Aug Mask #{i+1}")
        axes[1, i].axis('off')

    plt.tight_layout()
    axes[0, 1].axis('off')

    axes[1, 0].imshow(out_img)
    axes[1, 0].set_title("Augmented Image")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(out_mask, cmap='gray')
    axes[1, 1].set_title("Augmented Mask")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()
