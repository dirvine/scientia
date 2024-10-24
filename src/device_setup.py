import torch

def setup_device():
    """Set up and return the appropriate device (CPU/GPU/MPS)"""
    if torch.backends.mps.is_available():
        return torch.device("mps"), "Using Metal (MPS) backend."
    elif torch.cuda.is_available():
        return torch.device("cuda"), "Using CUDA backend."
    return torch.device("cpu"), "Using CPU."
