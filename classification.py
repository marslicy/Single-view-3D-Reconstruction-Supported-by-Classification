import torch
import torch.nn as nn
import torchvision

from data.shapenet import ShapeNetDataset
from data.shapenet_loader import ShapeNetDataLoader


class Model(nn.Module):
    def __init__(self, global_feature_size=128, local_feature_size=128, num_class=13):
        """
        Initialise the network

        Args:
            global_feature_size (int): The length of the global feature embeddings
            local_feature_size (int): The length of the local feature embeddings
        """
        super(Model, self).__init__()

        self.class_enc = torchvision.models.resnet50(pretrained=True)
        self.class_enc.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=global_feature_size),
            nn.ReLU(),
        )

        self.class_dec = nn.Sequential(  # might need to be modified
            nn.Linear(in_features=global_feature_size, out_features=num_class),
            nn.Sigmoid(),
        )

    def forward(self, x_in_class: torch.Tensor):
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
        pred_class = self.class_dec(class_emb)
        return pred_class


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


if __name__ == "__main__":
    config = {
        "experiment_name": "classification pretrain",
        "device": "cuda:0",  # or 'cuda:0 cpu'
        "batch_size": 256,
        "resume_ckpt": None,
        "learning_rate": 0.0001,
        "max_epochs": 50,
        "print_every_n": 100,
        "validate_every_n": 1000,
        "val_view": 1,
        "test_view": 1,
        "shape_num": 3,
        "global_feature_size": 128,
        "local_feature_size": 128,
        "num_class": 13,
    }

    train_dataset = ShapeNetDataset("train", config["val_view"], config["test_view"])
    train_dataloader = ShapeNetDataLoader(
        train_dataset,  # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config["batch_size"],  # The size of batches is defined here
        shape_num=config["shape_num"],
        shuffle=True,
    )
    model = Model(
        config["global_feature_size"], config["local_feature_size"], config["num_class"]
    )
    model = model.to(config["device"])
    loss_criterion = torch.nn.CrossEntropyLoss()
    loss_criterion.to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), config["learning_rate"])
    train_loss_running = 0.0

    for epoch in range(config["max_epochs"]):

        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            ShapeNetDataset.move_batch_to_device(batch, config["device"])
            x1 = batch["class"]
            y1 = batch["GT"]

            optimizer.zero_grad()

            # Perform forward pass
            pred = model(x1.float())

            loss = loss_criterion(pred, y1)

            # Backward
            loss.backward()
            # Update network parameters
            optimizer.step()
            # loss logging
            train_loss_running += loss.item()

            iteration = epoch * len(train_dataloader) + batch_idx

            if iteration % config["print_every_n"] == (config["print_every_n"] - 1):
                train_loss = train_loss_running / config["print_every_n"]

                print(f"[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss:.6f}")

                train_loss_running = 0.0
