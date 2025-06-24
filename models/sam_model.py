from segment_anything import sam_model_registry
import os
import urllib.request

CHECKPOINT_URLS = {
    'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'vit-l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
    'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
}

def download_checkpoint(model_type, checkpoint_path):
    url = CHECKPOINT_URLS.get(model_type)
    if url is None:
        raise ValueError(f"Unsupported model_type: {model_type}")
    if not os.path.exists(checkpoint_path):
        print(f"Downloading {model_type} checkpoint to {checkpoint_path}...")
        urllib.request.urlretrieve(url, checkpoint_path)
        print("Download complete.")

def load_sam_model(model_type='vit_h', checkpoint_path=None, device='cuda'):
    """
    Load a SAM model with vit_h or vit_b configuration.
    """
    if model_type not in CHECKPOINT_URLS:
        raise ValueError(f"Only 'vit_h' and 'vit_b' are supported. Got: {model_type}")
    if checkpoint_path is None:
        checkpoint_path = f"checkpoints/sam_{model_type}.pth"
    download_checkpoint(model_type, checkpoint_path)
    model = sam_model_registry[model_type](checkpoint=checkpoint_path)
    model = model.to(device)
    model.eval()
    return model