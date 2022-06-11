import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, global_feature_size: int, local_feature_size: int):
        """
        Initialise the network

        Args:
            global_feature_size (int): The length of the global feature embeddings
            local_feature_size (int): The length of the local feature embeddings
        """
        pass

    def forward(self, x_in_class: torch.Tensor, x_in_3d: torch.Tensor):
        """
        Prediction for the input

        Args:
            x_in_class (torch.Tensor): (B, 3, 127, 127) tensor. Input for extracting global features.
            x_in_3d (torch.Tensor): (B, 3, 127, 127) tensor. Input for extracting local features.

        Returns:
            pred_class (torch.Tensor): (B,) tensor. Classification results.
            pred_3d (torch.Tensor): (B, 32, 32, 32) tensor. Reconstructed voxels.
        """

        pass
        # return pred_class, pred_3d
