# Train.py

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data_loader.inpainting_dataset import InpaintingDataset
from models.sam_model import load_sam_model
from losses.dice_loss import DiceBCELoss, iou_score

DATA_ROOT = "/mnt/g/Authenta/data/authenta-inpainting-detection/dataset"
CHECKPOINT_PATH = "checkpoints/sam_vit_h.pth"
RESUME_CHECKPOINT = "best_model/sam_mask_decoder.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 25
LR = 3e-5
VAL_SPLIT = 0.2

# ------------------ Dataset & Loaders ------------------
dataset = InpaintingDataset(data_root=DATA_ROOT)
val_size = int(VAL_SPLIT * len(dataset))
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# ------------------ Model ------------------
sam = load_sam_model(model_type=MODEL_TYPE, checkpoint_path=CHECKPOINT_PATH)
sam = sam.to(device=DEVICE)
criterion = DiceBCELoss().to(DEVICE)
optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=LR)

# ------------------ Resume Logic ------------------
start_epoch = 0
if os.path.exists(RESUME_CHECKPOINT):
    print(f"✅ Loading checkpoint from {RESUME_CHECKPOINT}")
    checkpoint = torch.load(RESUME_CHECKPOINT, map_location=DEVICE)
    sam.mask_decoder.load_state_dict(checkpoint["mask_decoder"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    print(f"🔁 Resuming training from epoch {start_epoch}")
else:
    print("🆕 No checkpoint found, starting from scratch.")

# ------------------ Training Loop ------------------
for epoch in range(start_epoch, EPOCHS):
    sam.train()
    total_loss, total_iou = 0.0, 0.0
    sample_count = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=True)

    for batch in loop:
        imgs = batch['image'].to(DEVICE)
        masks_gt = batch['mask'].unsqueeze(1).to(DEVICE)
        points = batch['point'].to(DEVICE)

        B = imgs.size(0)
        optimizer.zero_grad()
        loss_batch, iou_batch = 0.0, 0.0

        for i in range(B):
            img = imgs[i].unsqueeze(0)
            mask_gt = masks_gt[i].unsqueeze(0)
            point = points[i].unsqueeze(0).unsqueeze(0)
            label = torch.tensor([[1]], device=DEVICE)

            with torch.no_grad():
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

            pred = F.interpolate(low_res_logits, size=mask_gt.shape[-2:], mode='bilinear', align_corners=False)

            loss = criterion(pred, mask_gt)
            iou = iou_score(pred, mask_gt)

            loss.backward()
            loss_batch += loss.item()
            iou_batch += iou.item()

        optimizer.step()
        total_loss += loss_batch
        total_iou += iou_batch
        sample_count += B
        loop.set_postfix(loss=loss_batch / B, iou=iou_batch / B)

    avg_loss = total_loss / sample_count
    avg_iou = total_iou / sample_count

    # ------------------ Validation ------------------
    sam.eval()
    val_iou_total = 0.0
    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=True)
        for batch in val_loop:
            img = batch['image'].to(DEVICE)
            mask_gt = batch['mask'].unsqueeze(1).to(DEVICE)
            point = batch['point'].unsqueeze(1).to(DEVICE)
            label = torch.ones((1, 1), device=DEVICE)

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

            pred = F.interpolate(low_res_logits, size=mask_gt.shape[-2:], mode='bilinear', align_corners=False)
            val_iou = iou_score(pred, mask_gt)
            val_iou_total += val_iou.item()
            val_loop.set_postfix(iou=val_iou.item())

    avg_val_iou = val_iou_total / len(val_loader.dataset)
    tqdm.write(f"✅ Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Train IoU: {avg_iou:.4f} | Val IoU: {avg_val_iou:.4f}")

    # ------------------ Save Checkpoint ------------------
    os.makedirs("best_model", exist_ok=True)
    torch.save({
        "mask_decoder": sam.mask_decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "train_iou": avg_iou,
        "val_iou": avg_val_iou,
    }, "best_model/sam_mask_decoder.pth")

    tqdm.write("✅ Model checkpoint saved at 'best_model/sam_mask_decoder.pth'")
    
    