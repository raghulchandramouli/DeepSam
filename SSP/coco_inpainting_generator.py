import os
import json
import random
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from diffusers import StableDiffusionInpaintPipeline
import urllib.request
import zipfile
from glob import glob

class COCOInpaintingGenerator:
    """
    A framework for generating a balanced dataset of Inpainting Images
    
    This class downloads COCO images, generates random masks, and creates inpainted
    versions using Stable Diffusion to address class imbalance in inpainting detection.
    """
    def __init__(self, real_images_path, output_dir="SSP", use_sd=True):
        self.real_images_path = real_images_path
        self.output_dir = output_dir
        self.use_sd = use_sd
        
        # Create directories
        self.real_images_dir = os.path.join(output_dir, "real-images")
        self.masks_dir = os.path.join(output_dir, "masks")
        self.inpainting_dir = os.path.join(output_dir, "inpainting")
        
        os.makedirs(self.real_images_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)
        os.makedirs(self.inpainting_dir, exist_ok=True)
        
        # Initialize Stable Diffusion pipeline
        if self.use_sd:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16
            )
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
    
    def download_coco_val(self):
        coco_dir = "coco_val2017"
        if os.path.exists(coco_dir):
            return coco_dir
            
        print("Downloading COCO validation images...")
        url = "http://images.cocodataset.org/zips/val2017.zip"
        zip_path = "val2017.zip"
        
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        os.remove(zip_path)
        os.rename("val2017", coco_dir)
        return coco_dir
        
    def generate_random_mask(self, size, mask_ratio=0.3):
        """Generate random irregular mask"""
        h, w = size
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Random shapes
        for _ in range(random.randint(3, 8)):
            points = [(random.randint(0, w), random.randint(0, h)) for _ in range(random.randint(3, 6))]
            img = Image.fromarray(mask)
            draw = ImageDraw.Draw(img)
            draw.polygon(points, fill=255)
            mask = np.array(img)
        
        return mask
    
    def create_inpainted_image(self, image, mask):
        """Inpainting using Stable Diffusion or noise fallback"""
        if self.use_sd:
            try:
                img_resized = image.resize((512, 512))
                mask_resized = Image.fromarray(mask).resize((512, 512))
                result = self.pipe(
                    prompt="can you fill in the white region with most naturally looking picture",
                    image=img_resized,
                    mask_image=mask_resized,
                    num_inference_steps=20,
                    guidance_scale=7.5
                ).images[0]
                return result.resize(image.size)
            except Exception as e:
                print(f"SD inpainting failed: {e}, using noise fallback")
        
        img_array = np.array(image)
        mask_bool = mask > 127
        noise = np.random.randint(0, 255, img_array.shape, dtype=np.uint8)
        img_array[mask_bool] = noise[mask_bool]
        return Image.fromarray(img_array)
    
    def generate_dataset(self, num_samples=10000, balance_ratio=0.5):
        """Generate balanced dataset with masked and non-masked samples"""
        # Gather up to num_samples images
        all_files = glob(os.path.join(self.real_images_path, "*.jpg"))
        img_files = all_files[:num_samples]
        num_files = len(img_files)
        # Determine how many to mask
        masked_count = int(num_files * balance_ratio)
        # Randomly pick which indices to mask
        mask_indices = set(random.sample(range(num_files), masked_count))
        
        for i, img_path in enumerate(img_files):
            if not os.path.exists(img_path):
                continue
                
            image = Image.open(img_path).convert('RGB')
            # Save original image
            real_img_path = os.path.join(self.real_images_dir, f"{i:06d}.jpg")
            image.save(real_img_path)
            
            if i in mask_indices:
                # Generate random mask
                mask = self.generate_random_mask(image.size[::-1])
                mask_img = Image.fromarray(mask)
                # Save mask
                mask_path = os.path.join(self.masks_dir, f"{i:06d}.png")
                mask_img.save(mask_path)
                # Create inpainted image
                inpainted = self.create_inpainted_image(image, mask)
                inpainted_path = os.path.join(self.inpainting_dir, f"{i:06d}.jpg")
                inpainted.save(inpainted_path)
            else:
                # No mask - save empty mask for consistency
                empty_mask = Image.fromarray(np.zeros(image.size[::-1], dtype=np.uint8))
                mask_path = os.path.join(self.masks_dir, f"{i:06d}.png")
                empty_mask.save(mask_path)
                # Copy original as "inpainted"
                inpainted_path = os.path.join(self.inpainting_dir, f"{i:06d}.jpg")
                image.save(inpainted_path)

if __name__ == "__main__":
    generator = COCOInpaintingGenerator(
        real_images_path="/mnt/g/Authenta/data/authenta-remaskable-inpainting-detection/balanced_images/real-images",
        output_dir="/mnt/g/Authenta/data/authenta-remaskable-inpainting-detection/balanced_images",
        use_sd=True
    )
    generator.generate_dataset(num_samples=10000, balance_ratio=0.5)