import torch
import torch.nn as nn
import torchvision


class Model(nn.Module):
    def __init__(self, global_feature_size=128, local_feature_size=128, num_class=13):
        """
        Initialise the network

        Args:
            global_feature_size (int): The length of the global feature embeddings
            local_feature_size (int): The length of the local feature embeddings
        """
        super(Model, self).__init__()
        self.view_enc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(negative_slope=0.01),
            View((-1, 256)),
            nn.Linear(in_features=256, out_features=local_feature_size),
            nn.ReLU(),
        )

        self.class_enc = torchvision.models.resnet18(pretrained=True)
        self.class_enc.fc = nn.Linear(in_features=512, out_features=global_feature_size)

        self.class_dec = nn.Sequential(  # might need to be modified
            nn.Linear(in_features=global_feature_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=num_class),
            nn.ReLU(),
        )

        self.shape_dec = nn.Sequential(
            nn.Linear(
                in_features=global_feature_size + local_feature_size, out_features=8192
            ),
            nn.ReLU(),
            View((-1, 8192, 1, 1, 1)),
            nn.ConvTranspose3d(
                in_channels=8192, out_channels=128, kernel_size=5, stride=2
            ),
            nn.ConvTranspose3d(
                in_channels=128,
                out_channels=64,
                kernel_size=5,
                stride=2,
                output_padding=1,
            ),
            nn.ConvTranspose3d(
                in_channels=64,
                out_channels=1,
                kernel_size=5,
                stride=2,
                output_padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x_in_class: torch.Tensor, x_in_3d: torch.Tensor):
        """
        Prediction for the input

        Args:
            x_in_class (torch.Tensor): (B, 3, 127, 127) tensor. Input for extracting global features.
            x_in_3d (torch.Tensor): (B, 3, 127, 127) tensor. Input for extracting local features.

        Returns:
            pred_class (torch.Tensor): (B, num_class) tensor. Classification results.
            pred_3d (torch.Tensor): (B, 32, 32, 32) tensor. Reconstructed voxels.
        """

        class_emb = self.class_enc(x_in_class)
        view_emb = self.view_enc(x_in_3d)
        pred_class = self.class_dec(class_emb)
        pred_3d = self.shape_dec(torch.cat((class_emb, view_emb), dim=1)).squeeze(1)
        return pred_class, pred_3d


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
