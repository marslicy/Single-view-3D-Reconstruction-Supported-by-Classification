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
        # Layer Definition
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg = nn.Sequential(*list(vgg16_bn.features.children()))[:27]
        # Don't update params in VGG16
        for param in vgg16_bn.parameters():
            param.requires_grad = False

        self.view_enc = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ELU(),
            # nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            nn.Conv2d(512, local_feature_size, kernel_size=2),
            nn.BatchNorm2d(local_feature_size),
            nn.ELU(),
        )

        self.class_enc = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            # torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            # torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(512, global_feature_size, kernel_size=2),
            torch.nn.BatchNorm2d(global_feature_size),
            torch.nn.ELU(),
        )

        self.class_dec = nn.Sequential(  # might need to be modified
            torch.nn.Conv2d(global_feature_size, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ELU(),
            torch.nn.Conv2d(64, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ELU(),
            torch.nn.Conv2d(16, 8, kernel_size=3),
            torch.nn.BatchNorm2d(8),
            torch.nn.ELU(),
            View((-1, 32)),
            nn.Linear(in_features=32, out_features=num_class),
            nn.Sigmoid(),
        )

        self.shape_dec = nn.Sequential(
            View((-1, (global_feature_size + local_feature_size) * 8, 2, 2, 2)),
            torch.nn.ConvTranspose3d(
                (global_feature_size + local_feature_size) * 8,
                512,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1),
            torch.nn.Sigmoid(),
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

        class_features = self.vgg(x_in_class)
        class_emb = self.class_enc(class_features)
        view_features = self.vgg(x_in_3d)
        view_emb = self.view_enc(view_features)
        pred_class = self.class_dec(class_emb)
        pred_3d = self.shape_dec(torch.cat((class_emb, view_emb), dim=1)).squeeze(1)
        return pred_class, pred_3d


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
