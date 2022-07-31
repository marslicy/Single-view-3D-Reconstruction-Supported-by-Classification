import torch

from model.model import Model


class Inference:
    def __init__(
        self,
        experiment: str,
        device: torch.device,
    ):
        """
        Args:
            global_feature_size (int): The length of the global feature embeddings.
            local_feature_size (int): The length of the local feature embeddings.
            experiment (str): path to experiment folder for the trained model.
            device (torch.device): torch device where inference is run.
        """
        self.model = Model(128, 128, 13)
        self.model.load_state_dict(torch.load(experiment, map_location=device))
        self.model = self.model.to(device)
        self.device = device

    def get_model(self):
        """
        Returns:
            model (Model): trained model loaded from disk
        """
        return self.model

    def reconstruct(self, x_in_class: torch.Tensor, x_in_3d: torch.Tensor):
        """
        Reconstruct the 3D voxels for an image taking from a random view.

        Args:
            inputs (torch.Tensor): (B, 3, 127, 127) tensor. Input image for reconstruction.

        Returns:
            pred_class (torch.Tensor): (B,) tensor. Classification results.
            pred_3d (torch.Tensor): (B, 32, 32, 32) tensor. Reconstructed voxels.

        """
        x_in_class = x_in_class.to(self.device)
        x_in_3d = x_in_3d.to(self.device)
        with torch.no_grad():
            pred_class, pred_3d = self.model(x_in_class, x_in_3d)
            pred_3d[pred_3d > 0.4] = 1
            pred_3d[pred_3d <= 0.4] = 0
        return pred_class, pred_3d
