import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.utils as vutils
from datetime import datetime
from data_loader.inpainting_dataset import InpaintingDataset
from models.sam_model import load_sam_model
from torch.utils.data import DataLoader
import numpy as np

def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def visualize_triplet_and_save(sam, val_loader, output_dir="outputs/random-patch", device="cuda", num_samples=130):
    os.makedirs(output_dir, exist_ok=True)
    sam.eval()
    count = 0
    total_iou = 0
    num_valid_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            
            img = batch['image'].to(device)
            point = batch['point'].unsqueeze(1).to(device)
            label = torch.ones((1, 1), device=device)

            has_mask = 'mask' in batch and batch['mask'] is not None

            # Encode image and prompt
            img_embed = sam.image_encoder(img)
            sparse_embed, dense_embed = sam.prompt_encoder(
                points=(point, label), boxes=None, masks=None
            )

            # Always generate prediction
            low_res_logits, _ = sam.mask_decoder(
                image_embeddings=img_embed,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                multimask_output=False
            )
            
            # Use image size for interpolation if no ground truth mask
            target_size = batch['mask'].shape[-2:] if has_mask else img.shape[-2:]
            pred_mask = F.interpolate(low_res_logits, size=target_size, mode='bilinear', align_corners=False)
            pred_mask = torch.sigmoid(pred_mask).squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(float)

            if has_mask:
                gt_mask = batch['mask'].unsqueeze(1).to(device)
                gt_mask_np = gt_mask.squeeze().cpu().numpy()
                # Calculate IoU
                iou_score = calculate_iou(pred_mask, gt_mask_np)
                total_iou += iou_score
                num_valid_samples += 1
                print(f"Sample {count} IoU Score: {iou_score:.4f}")
            else:
                gt_mask_np = None
                iou_score = None

            # Convert data for plotting
            img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()

            if has_mask:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img_np)
                axes[0].set_title("Input Image")
                axes[1].imshow(gt_mask_np, cmap="gray")
                axes[1].set_title("Ground Truth Mask")
                axes[2].imshow(pred_mask, cmap="gray")
                axes[2].set_title(f"Predicted Mask\nIoU: {iou_score:.4f}")
            else:
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(img_np)
                axes[0].set_title("Input Image")
                axes[1].imshow(pred_mask, cmap="gray")
                axes[1].set_title("Predicted Mask")
            
            for ax in axes:
                ax.axis('off')

            plt.tight_layout()

            # Save with timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = os.path.join(output_dir, f"triplet_{count:03d}_{timestamp}.png")
            plt.savefig(out_path)
            # Clean up plots
            plt.close('all')


            print(f" Saved: {out_path}")

            count += 1
            if count >= num_samples:
                break

        # Print aggregated statistics
        if num_valid_samples > 0:
            avg_iou = total_iou / num_valid_samples
            print("\n=== Final Statistics ===")
            print(f"Average IoU Score: {avg_iou:.4f}")
            print(f"Total samples processed: {count}")
            print(f"Samples with valid masks: {num_valid_samples}")
            
            # Save statistics to file
            stats_file = os.path.join(output_dir, "iou_statistics.txt")
            with open(stats_file, 'w') as f:
                f.write(f"Average IoU Score: {avg_iou:.4f}\n")
                f.write(f"Total samples processed: {count}\n")
                f.write(f"Samples with valid masks: {num_valid_samples}\n")

def get_image_files_recursive(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files

if __name__ == "__main__":
    # --- Config ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMAGE_DIR = "/mnt/g/Authenta/data/authenta-inpainting-detection/patch_crop/output_dir"
    MASK_DIR = "/mnt/g/Authenta/data/authenta-inpainting-detection/patch_crop/mask_tile"
    BASE_SAM_CKPT = "checkpoints/sam_vit_h_4b8939.pth"
    TRAINED_DECODER_PATH = "best_model_single_masks/sam_mask_decoder.pth"
    MODEL_TYPE = "vit_h"
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs/random_patch")
    NUM_SAMPLES = 130
    BATCH_SIZE = 1

    # --- Validate paths ---
    print(f"Checking IMAGE_DIR: {IMAGE_DIR}")
    print(f"Directory contents:")
    if os.path.exists(IMAGE_DIR):
        print(os.listdir(IMAGE_DIR)[:5])
    
    print(f"\nChecking MASK_DIR: {MASK_DIR}")
    print(f"Directory contents:")
    if os.path.exists(MASK_DIR):
        print(os.listdir(MASK_DIR)[:5])

    if MASK_DIR and not os.path.exists(MASK_DIR):
        print(f"Warning: Mask directory not found: {MASK_DIR}")
        MASK_DIR = None
    elif MASK_DIR:
        # Look for any mask files without requiring prefix
        mask_files = [f for f in os.listdir(MASK_DIR) 
                     if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(mask_files)} mask files")
        if mask_files:
            print("First few mask files:", mask_files[:3])
            # Check corresponding images exist by matching numbers
            base_names = [os.path.splitext(f)[0] for f in mask_files]
            image_files = os.listdir(IMAGE_DIR)
            matching_images = []
            for mask in mask_files:
                mask_num = ''.join(filter(str.isdigit, mask))
                for img in image_files:
                    if mask_num in img:
                        matching_images.append(img)
                        break
            print(f"Found {len(matching_images)} matching image-mask pairs")
            if matching_images:
                print("First few matches:", list(zip(matching_images[:3], mask_files[:3])))


    # Set SAM's required image size
    dataset = InpaintingDataset(
        image_dir=IMAGE_DIR, 
        mask_dir=MASK_DIR,
        target_size=(1024, 1024),  # SAM requires 1024x1024 images
        padding=True  # Enable padding to maintain aspect ratio
    )
    
    if len(dataset) == 0:
        raise ValueError(f"No valid images found in {IMAGE_DIR}")
    
    print(f"Found {len(dataset)} images")
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    sam = load_sam_model(model_type=MODEL_TYPE, checkpoint_path=BASE_SAM_CKPT).to(DEVICE)

    # Load trained decoder weights if they exist
    if os.path.exists(TRAINED_DECODER_PATH):
        checkpoint = torch.load(TRAINED_DECODER_PATH, map_location=DEVICE)
        sam.mask_decoder.load_state_dict(checkpoint["mask_decoder"])
    else:
        print(f"Warning: Trained decoder weights not found at {TRAINED_DECODER_PATH}")
        print("Using default weights...")

    # --- Run Visualization ---
    visualize_triplet_and_save(
        sam=sam,
        val_loader=loader,
        output_dir=OUTPUT_DIR,
        device=DEVICE,
        num_samples=NUM_SAMPLES
    )