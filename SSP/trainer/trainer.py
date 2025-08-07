import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class InpaintingDetectionTrainer:
    """Trainer for inpainting detection model"""
    
    def __init__(self, model, train_dataset, val_dataset, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        self.train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
        self.val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=1)
        
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            real = batch['real'].to(self.device)
            inpainted = batch['inpainted'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(real, inpainted)
            loss = self.criterion(pred, mask)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                real = batch['real'].to(self.device)
                inpainted = batch['inpainted'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                pred = self.model(real, inpainted)
                loss = self.criterion(pred, mask)
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self, epochs=25, save_dir="checkpoints"):
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                
            torch.save(self.model.state_dict(), os.path.join(save_dir, "latest_model.pth"))
