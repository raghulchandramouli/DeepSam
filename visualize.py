import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.utils as vutils
from datetime import datetime
from data_loader.inpainting_dataset import InpaintingDataset
from models.sam_model import load_sam_model
from torch.utils.data import DataLoader

def visualize_triplet_and_save(sam, val_loader, output_dir="triplet_outputs_visualizer_single_masks", device="cuda", num_samples=10):
    os.makedirs(output_dir, exist_ok=True)
    sam.eval()
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            img = batch['image'].to(device)
            gt_mask = batch['mask'].unsqueeze(1).to(device)
            point = batch['point'].unsqueeze(1).to(device)
            label = torch.ones((1, 1), device=device)

            # Encode image and prompt
            img_embed = sam.image_encoder(img)
            sparse_embed, dense_embed = sam.prompt_encoder(
                points=(point, label), boxes=None, masks=None
            )
            low_res_logits, _ = sam.mask_decoder(
                image_embeddings=img_embed,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                multimask_output=False
            )
            pred_mask = F.interpolate(low_res_logits, size=gt_mask.shape[-2:], mode='bilinear', align_corners=False)
            pred_mask = torch.sigmoid(pred_mask).squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(float)

            # Convert data for plotting
            img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
            gt_mask_np = gt_mask.squeeze().cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np)
            axes[0].set_title("Inpainted Image")

            axes[1].imshow(gt_mask_np, cmap="gray")
            axes[1].set_title("Ground Truth Mask")

            axes[2].imshow(pred_mask, cmap="gray")
            axes[2].set_title("Predicted Mask")

            for ax in axes:
                ax.axis('off')

            plt.tight_layout()

            # Save with timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = os.path.join(output_dir, f"triplet_{count:03d}_{timestamp}.png")
            plt.savefig(out_path)
            plt.close()

            print(f"✅ Saved: {out_path}")

            count += 1
            if count >= num_samples:
                break

if __name__ == "__main__":
    # --- Config ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_ROOT = "/mnt/g/Authenta/data/authenta-inpainting-detection/single_mask"
    BASE_SAM_CKPT = "checkpoints/sam_vit_h_4b8939.pth"
    TRAINED_DECODER_PATH = "best_model_vit_h/sam_mask_decoder.pth"
    MODEL_TYPE = "vit_h"
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "triplet_outputs_visualizer_single_masks")
    NUM_SAMPLES = 100

    # --- Load Data & Model ---
    dataset = InpaintingDataset(data_root=DATA_ROOT)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    sam = load_sam_model(model_type=MODEL_TYPE, checkpoint_path=BASE_SAM_CKPT).to(DEVICE)

    # Load trained decoder weights
    checkpoint = torch.load(TRAINED_DECODER_PATH, map_location=DEVICE)
    sam.mask_decoder.load_state_dict(checkpoint["mask_decoder"])

    # --- Run Visualization ---
    visualize_triplet_and_save(
        sam=sam,
        val_loader=loader,
        output_dir=OUTPUT_DIR,
        device=DEVICE,
        num_samples=NUM_SAMPLES
    )