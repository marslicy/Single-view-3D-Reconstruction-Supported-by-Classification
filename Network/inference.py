import torch


class Inference:
    def __init__(
        self,
        global_feature_size: int,
        local_feature_size: int,
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
        pass

    def get_model(self):
        """
        Returns:
            model (Model): trained model loaded from disk
        """
        pass
        # return model

    def reconstruct(self, input: torch.Tensor, target: torch.Tensor):
        """
        Reconstruct the 3D voxels for an image taking from a random view.

        Args:
            input (torch.Tensor): (B, 3, 127, 127) tensor. Input image for reconstruction.
            target (torch.Tensor): (B, 32, 32, 32) tensor. Ground truth.

        Returns:
            pred_class (torch.Tensor): (B,) tensor. Classification results.
            pred_3d (torch.Tensor): (B, 32, 32, 32) tensor. Reconstructed voxels.

        """
        pass
        # return pred_class, pred_3d
