import torch


from modules.signature_verification.model import SigNet


def load_signet_model(model_path: str):
    """
    Load SigNet model with strict checkpoint handling
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint format")

    # Handle common SigNet formats
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # fallback (legacy SigNet)

    model = SigNet()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, device
