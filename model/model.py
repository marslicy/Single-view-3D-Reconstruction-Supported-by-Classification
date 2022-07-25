from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, global_feature_size=128, local_feature_size=128, num_class=13):
        """
        Initialise the network

        Args:
            global_feature_size (int): The length of the global feature embeddings
            local_feature_size (int): The length of the local feature embeddings
        """
        super(Model, self).__init__()
        self.class_enc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(96),
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(256),
            View((-1, 256)),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
        )
        self.view_enc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(96),
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(256),
            View((-1, 256)),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
        )

        self.shape_dec = nn.Sequential(
            nn.Linear(in_features=128, out_features=8192),
            nn.ReLU(),
            View((-1, 128, 4, 4, 4)),
            ConvTranspose3dSame(
                in_channels=128,
                out_channels=128,
                kernel_size=5,
                stride=2,
            ),
            nn.BatchNorm3d(128),
            ConvTranspose3dSame(
                in_channels=128,
                out_channels=64,
                kernel_size=5,
                stride=2,
            ),
            nn.BatchNorm3d(64),
            ConvTranspose3dSame(
                in_channels=64,
                out_channels=1,
                kernel_size=5,
                stride=2,
            ),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

        self.class_dec = nn.Sequential(  # might need to be modified
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(in_features=32, out_features=13),
            nn.ReLU(),
            nn.BatchNorm1d(13),
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
        pred_3d = self.shape_dec(class_emb + view_emb).squeeze(1)
        return pred_class, pred_3d


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Conv3dSame(torch.nn.Conv3d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i0, i1, i2 = x.size()[-3:]

        pad_0 = self.calc_same_pad(
            i=i0, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_1 = self.calc_same_pad(
            i=i1, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )
        pad_2 = self.calc_same_pad(
            i=i2, k=self.kernel_size[2], s=self.stride[2], d=self.dilation[2]
        )

        if pad_0 > 0 or pad_1 > 0 or pad_2 > 0:
            x = F.pad(
                x,
                [
                    pad_2 // 2,
                    pad_2 - pad_2 // 2,
                    pad_1 // 2,
                    pad_1 - pad_1 // 2,
                    pad_0 // 2,
                    pad_0 - pad_0 // 2,
                ],
            )
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ConvTranspose3dSame(torch.nn.ConvTranspose3d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((i - 1) * s + d * k + 1 - i * s, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i0, i1, i2 = x.size()[-3:]

        pad_0 = self.calc_same_pad(
            i=i0, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_1 = self.calc_same_pad(
            i=i1, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )
        pad_2 = self.calc_same_pad(
            i=i2, k=self.kernel_size[2], s=self.stride[2], d=self.dilation[2]
        )
        x = F.conv_transpose3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if pad_0 > 0 or pad_1 > 0 or pad_2 > 0:
            x = F.pad(
                x,
                [
                    -(pad_2 // 2),
                    -pad_2 + pad_2 // 2,
                    -(pad_1 // 2),
                    -pad_1 + pad_1 // 2,
                    -(pad_0 // 2),
                    -pad_0 + pad_0 // 2,
                ],
            )
        return x
