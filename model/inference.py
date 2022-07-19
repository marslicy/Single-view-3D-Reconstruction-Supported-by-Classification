import numpy as np
import torch

from model.model import Model


class Inference:
    def __init__(self, device: str, ckpt: str):
        """
        Args:
            global_feature_size (int): The length of the global feature embeddings.
            local_feature_size (int): The length of the local feature embeddings.
            experiment (str): path to experiment folder for the trained model.
            device (torch.device): torch device where inference is run.
        """
        # load model and ckpts to the device and set to evaluation mode.
        self.model = Model()
        self.model.load_state_dict(torch.load(ckpt))
        self.model.to(device)
        self.model.eval()
        self.device = device

    def reconstruct(self, input: torch.Tensor):
        """
        Reconstruct the 3D voxels for an image taking from a random view.

        Args:
            input (torch.Tensor): (B, 3, 127, 127) tensor. Input image for reconstruction.
        Returns:
            pred_3d (torch.Tensor): (B, 32, 32, 32) tensor. Reconstructed voxels.

        """
        empty_prior = torch.tensor(
            np.empty((np.shape(input)[0], 1, 32, 32, 32)),
            device=self.device,
            dtype=torch.float32,
        )

        prediction = self.model(empty_prior, input)

        return prediction
        # return pred_class, pred_3d

    def write_binvox_file(self, input: torch.Tensor, filename: str):
        """
        Export the shape into a voxel file for better visualization alternatives

        Args:
            input (torch.Tensor): (32,32,32) tensor
            filename (str): the filename we should write into. should has the extension of ".binvox"

        """

        # convert the tensor back to numpy array
        # arr = input.cpu().detach().numpy()

        # write it to the voxel file
        pass
