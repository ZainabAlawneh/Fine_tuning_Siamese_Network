import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from modules.signature_verification.preprocess import preprocess_signature
from modules.signature_verification.model import SigNet
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


class SignatureEmbedder:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def extract(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # image_tensor is already batched [1, 1, H, W] from preprocessor
        # No need to unsqueeze again
        embedding = self.model(image_tensor.to(self.device))
        # Use L2 normalization: it makes Euclidean distance more stable
        return F.normalize(embedding, p=2, dim=1)


def cosine_similarity(e1: torch.Tensor, e2: torch.Tensor) -> float:
    """
    Cosine similarity between two normalized embeddings
    """
    return torch.sum(e1 * e2).item()

def verify_signature(
    test_embedding: torch.Tensor,
    reference_embeddings: list[torch.Tensor],
    threshold: float
) -> dict:
    """
    Compare test signature against multiple reference signatures
    """
    scores = [cosine_similarity(test_embedding, ref) for ref in reference_embeddings]

    best_score = max(scores)
    is_genuine = best_score >= threshold

    return {
        "best_similarity": best_score,
        "is_genuine": is_genuine,
        "threshold": threshold
    }

# 1. FIX: Updated Preprocessor to match Sigver (0 to 1 range, 150x220 size)
class SignaturePreprocessor:
    def __init__(self, target_size=(150, 220)):
        self.target_size = target_size  # (H, W)

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        # Re-using the logic from the example you found
        img_array = np.array(pil_img.convert("L"))

        # Note: Using your existing 'preprocess_signature' function logic here
        # or the one from the sigver integrated example
        processed = preprocess_signature(img_array, canvas_size=(952, 1360))

        # FIX: Correct range [0, 1] as required by the paper
        tensor = torch.from_numpy(processed).float().div(255)
        return tensor.view(1, 1, 150, 220)  # Shape: [Batch, Channel, H, W]


def load_signet_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SigNet()

    checkpoint = torch.load(model_path, map_location=device)

    # Sigver files are usually a tuple: (state_dict, optimizer, epoch)
    if isinstance(checkpoint, (tuple, list)):
        state_dict = checkpoint[0]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, device


# 3. FIX: Use Euclidean Distance (L2) instead of Cosine Similarity
# The paper authors suggest Euclidean distance for SigNet embeddings
# def verify_signature(test_emb, ref_embs, threshold=15.0):  # Threshold varies by dataset
#     # SigNet embeddings are usually NOT normalized for Euclidean comparison
#     distances = [torch.dist(test_emb, ref).item() for ref in ref_embs]
#     best_dist = min(distances)
#
#     # For distance: LOWER is better.
#     return {
#         "best_distance": best_dist,
#         "is_genuine": best_dist < threshold,
#         "threshold": threshold
#     }




