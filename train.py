# Train.py

import os
import torch
import torch.nn.Functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data_loader.inpainting_dataset import InpaintingDataset
from models.sam_model import load_sam_model
from losses.dice_loss import DiceBCELossm, iou_score

# Configs:
DATA_ROOT = "/mnt/g/Authenta/data/authenta-inpainting-detection/dataset"
CHECKPOINT_PATH = "sam_vit_b.pth"
MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
VAL_SPLIT = 0.2

# Loader & Model
dataset = InpaintingDataset(data_root=DATA_ROOT)
val_size = int(VAL_SPLIT * len(dataset))
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

sam = load_sam_model(model_type=MODEL_TYPE, checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=LR)

