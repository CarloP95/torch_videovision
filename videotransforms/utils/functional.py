import torch

def normalize(tensor, mean, std):
    """
    Args:
        tensor (Tensor): Tensor to normalize

    Returns:
        Tensor: Normalized tensor
    """
    try:
        res = (tensor.numpy() - mean) / std
    except ValueError as _:
        res = (tensor.numpy().transpose(1, 2, 3, 0) - mean) / std
        res = res.transpose(3, 0, 1, 2)
    #tensor.sub_(mean).div_(std)

    return torch.FloatTensor(res)
