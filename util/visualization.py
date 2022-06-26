import k3d
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_3d(voxels_pred: torch.Tensor, voxels_target: torch.Tensor, name: str):
    """
    Visualize the input 3D shape that presents one 3D shape as occupancy grid.

    Args:
        voxels_pred (torch.Tensor): (32, 32, 32) tensor. Output of the model.
        voxels_target (torch.Tensor): (32, 32, 32) tensor. Ground truth.
        name (str): name of the plot
    """
    point_pred = np.concatenate(
        [c[:, np.newaxis] for c in np.where(voxels_pred)], axis=1
    )
    point_target = np.concatenate(
        [c[:, np.newaxis] for c in np.where(voxels_target)], axis=1
    )
    plot = k3d.plot(
        name=name, grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55)
    )
    plot += k3d.points(positions=point_pred.astype(np.float32), color=0xD0D0D0)
    plot += k3d.points(
        positions=point_target.astype(np.float32), color=0x000000, opacity=0.25
    )
    plot.display()


def visualize_2d(image: torch.Tensor):
    """
    Visualize the input 2D image.

    Args:
        image (torch.Tensor): (3, 127, 127) tensor. Input of the model.
    """
    image = image.permute(1, 2, 0)
    plt.imshow(image)
