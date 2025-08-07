# train.py

import os
import random
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from models.sam_model import load_sam_model
from losses.dice_loss import DiceBCELoss, iou_score

# --------- Configurations ---------
IMAGE_DIR = "/mnt/g/Authenta/data/authenta-inpainting-detection/patch_crop/image_tile"
MASK_DIR  = "/mnt/g/Authenta/data/authenta-inpainting-detection/patch_crop/mask_tile"
CHECKPOINT = "checkpoints/sam_vit_h_4b8939.pth"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS     = 15
LR         = 3e-5
VAL_SPLIT  = 0.2
SAVE_DIR   = "best_model"

# --------- Helper Functions ---------
def match_pairs(img_dir: str, mask_dir: str):
    """
    Scans `img_dir` and `mask_dir`, pairs files whose names share the same numeric ID.
    E.g. "000123.jpg" pairs with "mask_000123.png".
    Returns two lists: image_paths, mask_paths in matching order.
    """
    imgs = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png")))
    pairs = []
    for img in imgs:
        key = "".join(filter(str.isdigit, img))
        if not key:
            continue
        for m in os.listdir(mask_dir):
            if key in m:
                pairs.append((os.path.join(img_dir, img), os.path.join(mask_dir, m)))
                break

    # Unzip into two lists
    image_paths, mask_paths = zip(*pairs)
    return list(image_paths), list(mask_paths)


class InpaintingDataset(Dataset):
    """
    Minimal Dataset for paired image & mask loading.
    - Resizes images with BILINEAR interpolation.
    - Resizes masks with NEAREST interpolation (to preserve binary values).
    """
    def __init__(self, image_paths, mask_paths, size=(1024, 1024)):
        assert len(image_paths) == len(mask_paths), "Image/mask count mismatch"
        self.images = image_paths
        self.masks  = mask_paths
        self.resize_img  = transforms.Resize(size, interpolation=Image.BILINEAR)
        self.resize_mask = transforms.Resize(size, interpolation=Image.NEAREST)
        self.to_tensor   = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load and preprocess image
        img = Image.open(self.images[idx]).convert("RGB")
        img = self.resize_img(img)

        # Load and preprocess mask
        msk = Image.open(self.masks[idx]).convert("L")
        msk = self.resize_mask(msk)

        # Convert to tensors
        return self.to_tensor(img), self.to_tensor(msk)


# --------- Main Training Pipeline ---------
def main():
    # 1) Match files
    image_files, mask_files = match_pairs(IMAGE_DIR, MASK_DIR)
    assert image_files and mask_files, "No image-mask pairs found!"
    print(f" Matched {len(image_files)} pairs")

    # 2) Create dataset and split
    dataset = InpaintingDataset(image_files, mask_files)
    val_n = int(VAL_SPLIT * len(dataset))
    train_n = len(dataset) - val_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)

    # 3) Load SAM model & training components
    sam       = load_sam_model("vit_h", CHECKPOINT).to(DEVICE)
    criterion = DiceBCELoss().to(DEVICE)
    optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=LR)

    best_val_iou = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # --- Training Phase ---
        sam.train()
        train_loss = 0.0

        for imgs, msks in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)  # [B,3,H,W], [B,1,H,W]
            optimizer.zero_grad()

            # Process batch in a single forward/backward pass
            losses = []
            for i in range(imgs.size(0)):
                img = imgs[i : i+1]
                msk = msks[i : i+1]
                # Prompt point at image center
                pt  = torch.tensor([[[msk.size(2)//2, msk.size(3)//2]]], device=DEVICE)
                lbl = torch.ones((1,1), device=DEVICE)

                # Encode image once (no grad)
                with torch.no_grad():
                    emb = sam.image_encoder(img)
                sp_emb, dn_emb = sam.prompt_encoder(points=(pt, lbl), boxes=None, masks=None)

                # Decode mask
                logits, _ = sam.mask_decoder(
                    image_embeddings=emb,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sp_emb,
                    dense_prompt_embeddings=dn_emb,
                    multimask_output=False,
                )
                pred = F.interpolate(logits, size=msk.shape[-2:], mode="bilinear", align_corners=False)

                # Compute loss
                loss = criterion(pred, msk)
                loss.backward()
                losses.append(loss.item())

            optimizer.step()
            train_loss += sum(losses) / len(losses)

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        sam.eval()
        total_iou = 0.0

        with torch.no_grad():
            for imgs, msks in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
                pt  = torch.tensor([[[msks.size(2)//2, msks.size(3)//2]]], device=DEVICE)
                lbl = torch.ones((1,1), device=DEVICE)

                emb = sam.image_encoder(imgs)
                sp_emb, dn_emb = sam.prompt_encoder(points=(pt, lbl), boxes=None, masks=None)

                logits, _ = sam.mask_decoder(
                    image_embeddings=emb,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sp_emb,
                    dense_prompt_embeddings=dn_emb,
                    multimask_output=False,
                )
                pred = F.interpolate(logits, size=msks.shape[-2:], mode="bilinear", align_corners=False)
                total_iou += iou_score(pred, msks).item()

        avg_val_iou = total_iou / len(val_loader)
        print(f"\n Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

        # --- Checkpointing ---
        ckpt = {
            "decoder": sam.mask_decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_iou": avg_val_iou,
        }
        torch.save(ckpt, os.path.join(SAVE_DIR, "latest.pth"))
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(ckpt, os.path.join(SAVE_DIR, "best.pth"))
            print("🏆 New best model saved!")

    print("  Training complete!")
    print(f" Best Val IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()
