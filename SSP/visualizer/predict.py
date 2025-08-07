import os
import sys
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Add the parent directory ('SSP') to the Python path to resolve the module import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import InpaintingDetectionModel

def preprocess_image(img_path, target_size=(512, 512)):
    """Loads and preprocesses an image."""
    image = Image.open(img_path).convert('RGB')
    image = image.resize(target_size, Image.Resampling.BILINEAR)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # Add batch dimension

def predict(model, real_img_path, inpainted_img_path, device="cuda"):
    """
    Runs inpainting detection on a pair of images.
    
    Args:
        model: The loaded InpaintingDetectionModel.
        real_img_path (str): Path to the original image. Can be None.
        inpainted_img_path (str): Path to the potentially inpainted image.
        device (str): The device to run inference on ('cuda' or 'cpu').
    
    Returns:
        A numpy array of the predicted mask.
    """
    model.eval()
    model.to(device)
    
    # If only one image is provided, use it for both inputs.
    if real_img_path is None:
        print("Warning: No 'real' image provided. Using the 'inpainted' image for both inputs.")
        real_tensor = preprocess_image(inpainted_img_path, target_size=(512, 512)).to(device)
    else:
        real_tensor = preprocess_image(real_img_path, target_size=(512, 512)).to(device)

    inpainted_tensor = preprocess_image(inpainted_img_path, target_size=(512, 512)).to(device)
    
    with torch.no_grad():
        output_mask = model(real_tensor, inpainted_tensor)
        
    # Post-process the output to a binary mask
    mask_np = output_mask.squeeze().cpu().numpy()
    return (mask_np > 0.5).astype(float)

def main():
    parser = argparse.ArgumentParser(description="Predict inpainting masks for an image.")
    parser.add_argument("--inpainted_image", type=str, required=True, help="Path to the potentially inpainted photo.")
    parser.add_argument("--real_image", type=str, default=None, help="Path to the original photo (optional). If not provided, the inpainted image is used for both inputs.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model weights (.pth file).")
    parser.add_argument("--output_path", type=str, default="prediction_output.png", help="Path to save the output visualization.")
    parser.add_argument("--cpu", action="store_true", help="Force use of CPU.")
    
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {DEVICE}")

    # --- Load Model ---
    if not os.path.exists(args.model_path):
        print(f"Error: Model weights not found at {args.model_path}")
        print("This script requires a trained model. You can train one using the provided scripts.")
        print("Proceeding with a randomly initialized model for demonstration purposes.")
        model = InpaintingDetectionModel(in_channels=6, out_channels=1)
    else:
        print(f"Loading model from {args.model_path}")
        model = InpaintingDetectionModel(in_channels=6, out_channels=1)
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))

    # --- Run Prediction ---
    predicted_mask = predict(model, args.real_image, args.inpainted_image, device=DEVICE)

    # --- Visualize Results ---
    inpainted_display = Image.open(args.inpainted_image).resize((512, 512))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(inpainted_display)
    axes[0].set_title("Input Photo")
    axes[0].axis('off')
    
    axes[1].imshow(predicted_mask, cmap='gray')
    axes[1].set_title("Predicted Inpainting Mask")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(args.output_path)
    plt.show()
    
    print(f"Prediction visualization saved to {args.output_path}")

if __name__ == '__main__':
    main()
