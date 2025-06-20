from segment_anything import sam_model_registry

def load_sam_model(model_type, checkpoint_path, device='cuda'):
    """
    Load a SAM model with specified configuration
    Args:
        model_type (str): Type of SAM model ('vit_h', 'vit_l', 'vit_b')
        checkpoint_path (str): Path to model checkpoint
        device (str): Device to load model on ('cuda' or 'cpu')
    Returns:
        model: Loaded SAM model
    """
    # Initialize model
    model = sam_model_registry[model_type](checkpoint=checkpoint_path)
    
    # Move to specified device
    model = model.to(device)
    model.eval()
    
    return model