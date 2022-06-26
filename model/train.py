from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.shapenet import ShapeNetDataset
from data.shapenet_loader import ShapeNetDataLoader
from model.model import Model


def train(
    model: Model,
    train_dataloader: DataLoader,
    val_dataloader_view: DataLoader,
    val_dataloader_shape: DataLoader,
    config: Dict,
):
    """
    Training process. Loss function for both classification and reconstruction task should be cross-entropy.
    In validation phase, it should validate for both unseen view and unseen shape.

    Args:
        model (Model): The being trained model.
        train_dataloader (DataLoader): Dataloader that provide training data.
        val_dataloader_view (DataLoader): Provide validation data whose shape was seen by the model,
                                          but with new views.
        val_dataloader_shape (DataLoader): Provide validation data whose shape was not seen by the model.
        config (Dict): configuration for training - has the following keys
                       'experiment_name': name of the experiment, checkpoint will be saved to folder "3dml-project/runs/<experiment_name>"
                       'device': device on which model is trained, e.g. 'cpu' or 'cuda:0'
                       'batch_size': batch size for training and validation dataloaders
                       'resume_ckpt': None if training from scratch, otherwise path to checkpoint (saved weights)
                       'learning_rate': learning rate for optimizer
                       'max_epochs': total number of epochs after which training should stop
                       'print_every_n': print train loss every n iterations
                       'validate_every_n': print validation loss and validation accuracy every n iterations
    """
    # Train classification model
    loss_criterion = torch.nn.CrossEntropyLoss()
    loss_criterion.to(config["device"])

    optimizer = torch.optim.Adam(model.parameters(), config["learning_rate"])

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.5
    )  # not necessary

    writer = SummaryWriter(f'runs/{config["experiment_name"]}/log')

    model.train()

    train_loss_running = 0.0
    train_loss_class_running = 0.0
    train_loss_3d_running = 0.0
    val_view_loss_running = 0.0
    val_view_loss_class_running = 0.0
    val_view_loss_3d_running = 0.0
    val_shape_loss_running = 0.0
    val_shape_loss_class_running = 0.0
    val_shape_loss_3d_running = 0.0
    train_best_loss = float("inf")
    val_view_best_loss = float("inf")
    val_shape_best_loss = float("inf")

    for epoch in range(config["max_epochs"]):

        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            ShapeNetDataset.move_batch_to_device(batch, config["device"])
            x1 = batch["class"]
            x2 = batch["encoder"]
            y1 = batch["GT"]
            y2 = batch["3D"]

            optimizer.zero_grad()

            # Perform forward pass
            pred_class, pred_3d = model(x1.float(), x2.float())

            loss_class = loss_criterion(pred_class, y1)
            loss_3d = loss_criterion(pred_3d, y2)
            loss = loss_class + loss_3d

            # Backward
            loss.backward()
            # Update network parameters
            optimizer.step()
            # loss logging
            train_loss_running += loss.item()
            train_loss_class_running += loss_class.item()
            train_loss_3d_running += loss_3d.item()

            iteration = epoch * len(train_dataloader) + batch_idx

            if iteration % config["print_every_n"] == (config["print_every_n"] - 1):
                train_loss = train_loss_running / config["print_every_n"]
                train_loss_class = train_loss_class_running / config["print_every_n"]
                train_loss_3d = train_loss_3d_running / config["print_every_n"]

                print(f"[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss:.6f}")
                print(
                    f"[{epoch:03d}/{batch_idx:05d}] train_loss_class: {train_loss_class:.6f}"
                )
                print(
                    f"[{epoch:03d}/{batch_idx:05d}] train_loss_3d: {train_loss_3d:.6f}"
                )

                writer.add_scalar(
                    "Training loss",
                    train_loss,
                    epoch * len(train_dataloader) + batch_idx,
                )
                writer.add_scalar(
                    "Training loss (class)",
                    train_loss_class,
                    epoch * len(train_dataloader) + batch_idx,
                )
                writer.add_scalar(
                    "Training loss (3D)",
                    train_loss_3d,
                    epoch * len(train_dataloader) + batch_idx,
                )

                # save best train model and latent codes
                if train_loss < train_best_loss:
                    torch.save(
                        model.state_dict(),
                        f'./runs/{config["experiment_name"]}/train_model_best.ckpt',
                    )
                    train_best_loss = train_loss

                train_loss_running = 0.0
                train_loss_class_running = 0.0
                train_loss_3d_running = 0.0

        # val_view
            if batch_idx == config["validate_every_n"]:
                model.eval()
                for batch_idx, batch in enumerate(val_dataloader_view):
                    # Move batch to device
                    ShapeNetDataset.move_batch_to_device(batch, config["device"])
                    x1 = batch["class"]
                    x2 = batch["encoder"]
                    y1 = batch["GT"]
                    y2 = batch["3D"]

                    optimizer.zero_grad()

                    # Perform forward pass
                    pred_class, pred_3d = model(x1.float(), x2.float())

                    loss_class = loss_criterion(pred_class, y1)
                    loss_3d = loss_criterion(pred_3d, y2)
                    loss = config["a"] * loss_class + config["b"] * loss_3d

                    # Backward
                    loss.backward()
                    # Update network parameters
                    optimizer.step()
                    # loss logging
                    val_view_loss_running += loss.item()
                    val_view_loss_class_running += loss_class.item()
                    val_view_loss_3d_running += loss_3d.item()

                    iteration = epoch * len(val_dataloader_view) + batch_idx

                    if iteration % config["print_every_n"] == (config["print_every_n"] - 1):
                        val_view_loss = val_view_loss_running / config["print_every_n"]
                        val_view_loss_class = (
                            val_view_loss_class_running / config["print_every_n"]
                        )
                        val_view_loss_3d = val_view_loss_3d_running / config["print_every_n"]

                        print(
                            f"[{epoch:03d}/{batch_idx:05d}] val_view_loss: {val_view_loss:.6f}"
                        )
                        print(
                            f"[{epoch:03d}/{batch_idx:05d}] val_view_loss_class: {val_view_loss_class:.6f}"
                        )
                        print(
                            f"[{epoch:03d}/{batch_idx:05d}] val_view_loss_3d: {val_view_loss_3d:.6f}"
                        )

                        writer.add_scalar(
                            "View validation loss",
                            val_view_loss,
                            epoch * len(val_dataloader_view) + batch_idx,
                        )
                        writer.add_scalar(
                            "View validation  loss (class)",
                            val_view_loss_class,
                            epoch * len(val_dataloader_view) + batch_idx,
                        )
                        writer.add_scalar(
                            "View validation  loss (3D)",
                            val_view_loss_3d,
                            epoch * len(val_dataloader_view) + batch_idx,
                        )

                        if val_view_loss < val_view_best_loss:
                            torch.save(
                                model.state_dict(),
                                f'./runs/{config["experiment_name"]}/val_view_model_best.ckpt',
                            )
                            val_view_best_loss = val_view_loss

                        val_view_loss_running = 0.0
                        val_view_loss_class_running = 0.0
                        val_view_loss_3d_running = 0.0

                # val_shape
                for batch_idx, batch in enumerate(val_dataloader_shape):
                    # Move batch to device
                    ShapeNetDataset.move_batch_to_device(batch, config["device"])
                    x1 = batch["class"]
                    x2 = batch["encoder"]
                    y1 = batch["GT"]
                    y2 = batch["3D"]

                    optimizer.zero_grad()

                    # Perform forward pass
                    pred_class, pred_3d = model(x1.float(), x2.float())

                    loss_class = loss_criterion(pred_class, y1)
                    loss_3d = loss_criterion(pred_3d, y2)
                    loss = loss_class + loss_3d

                    # Backward
                    loss.backward()
                    # Update network parameters
                    optimizer.step()
                    # loss logging
                    val_shape_loss_running += loss.item()
                    val_shape_loss_class_running += loss_class.item()
                    val_shape_loss_3d_running += loss_3d.item()

                    iteration = epoch * len(val_dataloader_shape) + batch_idx

                    if iteration % config["print_every_n"] == (config["print_every_n"] - 1):
                        val_shape_loss = val_shape_loss_running / config["print_every_n"]
                        val_shape_loss_class = (
                            val_shape_loss_class_running / config["print_every_n"]
                        )
                        val_shape_loss_3d = val_shape_loss_3d_running / config["print_every_n"]

                        print(
                            f"[{epoch:03d}/{batch_idx:05d}] val_shape_loss: {val_shape_loss:.6f}"
                        )
                        print(
                            f"[{epoch:03d}/{batch_idx:05d}] val_shape_loss_class: {val_shape_loss_class:.6f}"
                        )
                        print(
                            f"[{epoch:03d}/{batch_idx:05d}] val_shape_loss_3d: {val_shape_loss_3d:.6f}"
                        )

                        writer.add_scalar(
                            "Shape validation loss",
                            val_shape_loss,
                            epoch * len(val_dataloader_shape) + batch_idx,
                        )
                        writer.add_scalar(
                            "Shape validation  loss (class)",
                            val_shape_loss_class,
                            epoch * len(val_dataloader_shape) + batch_idx,
                        )
                        writer.add_scalar(
                            "Shape validation  loss (3D)",
                            val_shape_loss_3d,
                            epoch * len(val_dataloader_shape) + batch_idx,
                        )

                        if val_shape_loss < val_shape_best_loss:
                            torch.save(
                                model.state_dict(),
                                f'./runs/{config["experiment_name"]}/val_shape_model_best.ckpt',
                            )
                            val_shape_best_loss = val_shape_loss

                        val_shape_loss_running = 0.0
                        val_shape_loss_class_running = 0.0
                        val_shape_loss_class_running = 0.0

            # # visualize first 5 training shape reconstructions from latent codes
            # if iteration % config['visualize_every_n'] == (config['visualize_every_n'] - 1):
            #     # Set model to eval
            #     model.eval()
            #     latent_vectors_for_vis = latent_vectors(torch.LongTensor(range(min(5, latent_vectors.num_embeddings))).to(device))
            #     for latent_idx in range(latent_vectors_for_vis.shape[0]):
            #         # create mesh and save to disk
            #         evaluate_model_on_grid(model, latent_vectors_for_vis[latent_idx, :], device, 64, f'exercise_3/runs/{config["experiment_name"]}/meshes/{iteration:05d}_{latent_idx:03d}.obj')
            #     # set model back to train
            #     model.train()

        # lr scheduler update
        scheduler.step()


def test(
    model: Model,
    test_dataloader_view: DataLoader,
    test_dataloader_shape: DataLoader,
    config: Dict,
):
    """
    Test the model using both unseen view and unseen shape.

    Args:
        model (Model): The trained model.
        test_dataloader_view (DataLoader): Provide test data whose shape was seen by the model,
                                           but with new views.
        test_dataloader_shape (DataLoader): Provide test data whose shape was not seen by the model.

    Retruns:
        tbd
    """
    loss_criterion = torch.nn.CrossEntropyLoss()
    loss_criterion.to(config["device"])

    model.eval()

    test_view_loss_running = 0.0
    test_view_loss_class_running = 0.0
    test_view_loss_3d_running = 0.0
    test_shape_loss_running = 0.0
    test_shape_loss_class_running = 0.0
    test_shape_loss_3d_running = 0.0
    with torch.no_grad():

        for epoch in range(config["max_epochs"]):
            # test_view
            for batch_idx, batch in enumerate(test_dataloader_view):
                # Move batch to device
                ShapeNetDataset.move_batch_to_device(batch, config["device"])
                x1 = batch["class"]
                x2 = batch["encoder"]
                y1 = batch["GT"]
                y2 = batch["3D"]

                # Perform forward pass
                pred_class, pred_3d = model(x1.float(), x2.float())

                loss_class = loss_criterion(pred_class, y1)
                loss_3d = loss_criterion(pred_3d, y2)
                loss = loss_class + loss_3d

                # loss logging
                test_view_loss_running += loss.item()
                test_view_loss_class_running += loss_class.item()
                test_view_loss_3d_running += loss_3d.item()

            test_view_loss = test_view_loss_running / len(test_dataloader_view)
            test_view_loss_class = test_view_loss_class_running / len(
                test_dataloader_view
            )
            test_view_loss_3d = test_view_loss_3d_running / len(test_dataloader_view)

            print(f"[{epoch:03d}/{batch_idx:05d}] test_view_loss: {test_view_loss:.6f}")
            print(
                f"[{epoch:03d}/{batch_idx:05d}] test_view_loss_class: {test_view_loss_class:.6f}"
            )
            print(
                f"[{epoch:03d}/{batch_idx:05d}] test_view_loss_3d: {test_view_loss_3d:.6f}"
            )

            # test_shape
            for batch_idx, batch in enumerate(test_dataloader_shape):
                # Move batch to device
                ShapeNetDataset.move_batch_to_device(batch, config["device"])
                x1 = batch["class"]
                x2 = batch["encoder"]
                y1 = batch["GT"]
                y2 = batch["3D"]

                # Perform forward pass
                pred_class, pred_3d = model(x1.float(), x2.float())

                loss_class = loss_criterion(pred_class, y1)
                loss_3d = loss_criterion(pred_3d, y2)
                loss = loss_class + loss_3d

                # loss logging
                test_shape_loss_running += loss.item()
                test_shape_loss_class_running += loss_class.item()
                test_shape_loss_3d_running += loss_3d.item()

            test_shape_loss = test_shape_loss_running / len(test_dataloader_shape)
            test_shape_loss_class = test_shape_loss_class_running / len(
                test_dataloader_shape
            )
            test_shape_loss_3d = test_shape_loss_3d_running / len(test_dataloader_shape)

            print(
                f"[{epoch:03d}/{batch_idx:05d}] test_shape_loss: {test_shape_loss:.6f}"
            )
            print(
                f"[{epoch:03d}/{batch_idx:05d}] test_shape_loss_class: {test_shape_loss_class:.6f}"
            )
            print(
                f"[{epoch:03d}/{batch_idx:05d}] test_shape_loss_3d: {test_shape_loss_3d:.6f}"
            )


def main(config):
    """
    Function for training DeepSDF
        config (Dict): configuration for training - has the following keys
                       'experiment_name': name of the experiment, checkpoint will be saved to folder "3dml-project/runs/<experiment_name>"
                       'device': device on which model is trained, e.g. 'cpu' or 'cuda:0'
                       'batch_size': batch size for training and validation dataloaders
                       'resume_ckpt': None if training from scratch, otherwise path to checkpoint (saved weights)
                       'learning_rate': learning rate for optimizer
                       'max_epochs': total number of epochs after which training should stop
                       'print_every_n': print train loss every n iterations
                       'validate_every_n': print validation loss and validation accuracy every n iterations
                       'val_view'
                       'test_view'
    """

    # create dataloaders
    train_dataset = ShapeNetDataset("train", config["val_view"], config["test_view"])
    train_dataloader = ShapeNetDataLoader(
        train_dataset,  # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config["batch_size"],  # The size of batches is defined here
        shuffle=True,
    )

    val_dataset_view = ShapeNetDataset(
        "view_val", config["val_view"], config["test_view"]
    )
    val_dataloader_view = ShapeNetDataLoader(
        val_dataset_view,  # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config["batch_size"],  # The size of batches is defined here
        shuffle=True,
    )

    val_dataset_shape = ShapeNetDataset(
        "shape_val", config["val_view"], config["test_view"]
    )
    val_dataloader_shape = ShapeNetDataLoader(
        val_dataset_shape,  # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config["batch_size"],  # The size of batches is defined here
        shuffle=True,
    )

    test_dataset_view = ShapeNetDataset(
        "view_test", config["val_view"], config["test_view"]
    )
    test_dataloader_view = ShapeNetDataLoader(
        test_dataset_view,  # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config["batch_size"],  # The size of batches is defined here
    )

    test_dataset_shape = ShapeNetDataset(
        "shape_test", config["val_view"], config["test_view"]
    )
    test_dataloader_shape = ShapeNetDataLoader(
        test_dataset_shape,  # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config["batch_size"],  # The size of batches is defined here
    )

    # Instantiate model
    model = Model()

    # Move model to specified device
    model.to(config["device"])

    # Create folder for saving checkpoints
    Path(f'./runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Start training
    train(
        model,
        train_dataloader,
        val_dataloader_view,
        val_dataloader_shape,
        config,
    )

    test(
        model,
        test_dataloader_view,
        test_dataloader_shape,
        config,
    )
