import torch
def TensorNorm(tensor):

    tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor)) # 0-1 normalization

    return tensor