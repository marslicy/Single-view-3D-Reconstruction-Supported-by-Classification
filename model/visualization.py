import torch


def visualize_3d(voxels_pred: torch.Tensor, voxels_target: torch.Tensor):
    """
    Visualize the input 3D shape that presents one 3D shape as occupancy grid.

    Args:
        voxels_pred (torch.Tensor): (32, 32, 32) tensor. Output of the model.
        voxels_target (torch.Tensor): (32, 32, 32) tensor. Ground truth.
    """
    pass


def visualize_2d(image: torch.Tensor):
    """
    Visualize the input 2D image.

    Args:
        image (torch.Tensor): (3, 127, 127) tensor. Input of the model.
    """
    pass
