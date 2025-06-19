import torch
from segment_anything import sam_model_registry

def load_sam_model(model_type='vit_b', checkpoint_path=None):
    """
    Load a Segment Anything Model (SAM) from the specified checkpoint.
    
    Args:
        model_type (str): Type of SAM model to load. Default is 'vit_b'.
        checkpoint_path (str): Path to the model checkpoint. If None, uses the default model.
        
    Returns:
        sam_model: Loaded SAM model.
    """
    
    assert model_type in ['vit_b', 'vit_l', 'vit_h'], "Invalid model type. Choose from 'vit_b', 'vit_l', or 'vit_h'."
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path) 
    sam_model.to(device=device)
    return sam_model 