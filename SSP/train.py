import torch
from sklearn.model_selection import train_test_split
from dataset.dataset import InpaintingDetectionDataset
from model.model import InpaintingDetectionModel
from trainer.trainer import InpaintingDetectionTrainer

def main():
    # Configuration
    data_dir = "/mnt/g/Authenta/data/authenta-remaskable-inpainting-detection/balanced_images"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create full dataset
    full_dataset = InpaintingDetectionDataset(data_dir)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Initialize model
    model = InpaintingDetectionModel()
    
    # Initialize trainer
    trainer = InpaintingDetectionTrainer(model, train_dataset, val_dataset, device)
    
    # Train
    trainer.train(epochs=25, save_dir="SSP/checkpoints")

if __name__ == "__main__":
    main()
