import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.device("cpu")
    return torch.device("cpu"), torch.device("cpu")