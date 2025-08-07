import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet152

class InpaintingDetectionModel(nn.Module):
    """ResNet-152 encoder with U-Net decoder for inpainting detection segmentation"""
    
    def __init__(self, in_channels=6, out_channels=1):
        super().__init__()
        
        # ResNet-152 encoder
        resnet = resnet152(weights='IMAGENET1K_V1')
        
        # Modify first conv layer for 6 channels
        resnet.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        
        # Copy RGB weights to new channels
        with torch.no_grad():
            original_weight = resnet152(weights='IMAGENET1K_V1').conv1.weight
            resnet.conv1.weight[:, :3] = original_weight
            resnet.conv1.weight[:, 3:6] = original_weight
        
        # Encoder layers
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        
        # Decoder with upsampling
        self.decoder5 = self._decoder_block(2048, 1024)
        self.decoder4 = self._decoder_block(1024 + 1024, 512)
        self.decoder3 = self._decoder_block(512 + 512, 256)
        self.decoder2 = self._decoder_block(256 + 256, 128)
        self.decoder1 = self._decoder_block(128 + 64, 64)
        
        # --- MODIFIED ---
        # Final layer to produce the output mask. 
        # The input channel size is 64 from self.decoder1
        self.final = nn.Conv2d(64, out_channels, 1)
        
    def _decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, real, inpainted):
        # Concatenate inputs
        x = torch.cat([real, inpainted], dim=1)  # [B, 6, 512, 512]
        
        # Encoder with skip connections
        e1 = self.encoder1(x)    # [B, 64, 256, 256]
        e2 = self.encoder2(e1)   # [B, 256, 128, 128]
        e3 = self.encoder3(e2)   # [B, 512, 64, 64]
        e4 = self.encoder4(e3)   # [B, 1024, 32, 32]
        e5 = self.encoder5(e4)   # [B, 2048, 16, 16]
        
        # Decoder with skip connections and upsampling
        d5 = self.decoder5(e5)   # [B, 1024, 32, 32]
        d4 = self.decoder4(torch.cat([d5, e4], dim=1))  # [B, 512, 64, 64]
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))  # [B, 256, 128, 128]
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))  # [B, 128, 256, 256]
        d1 = self.decoder1(torch.cat([d2, e1], dim=1))  # [B, 64, 512, 512]
        
        # --- MODIFIED ---
        # Final output. d1 is already at the target resolution.
        # The output size will be [B, 1, 512, 512], matching the target mask.
        return torch.sigmoid(self.final(d1))