from datetime import datetime
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.shapenet import ShapeNetDataset
from model.model import Model


def train(
    model: Model,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
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
    #early stop
    last_loss = 100
    patience = 3
    trigger_times = 0

    loss_criterion = torch.nn.BCELoss()
    loss_criterion.to(config["device"])

    optimizer = torch.optim.Adadelta(model.parameters(), config["learning_rate"])
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    now = datetime.now()
    dt_string = now.strftime("%m-%d-%H-%M")
    writer = SummaryWriter(f'runs/{dt_string}-{config["experiment_name"]}/log')

    model.train()

    best_loss_val = np.inf

    train_loss_running = 0.0

    for epoch in range(config["max_epochs"]):
        #         print("-"*128)
        print("Epoch %d/%.0f" % (epoch + 1, config["max_epochs"]))
        for batch_idx, batch in enumerate(train_dataloader):

            ShapeNetDataset.move_batch_to_device(batch, config["device"])
            input_images = batch["Image"]
            input_voxels = batch["3D_prior"]
            target_voxels = batch["GT"]

            optimizer.zero_grad()

            for _ in range(config["iter"]):
                input_voxels = model(input_voxels, input_images)
            pred = input_voxels.squeeze(1)

            loss = loss_criterion(pred, target_voxels)
            loss.backward()
            optimizer.step()

            train_loss_running += loss.item()

            iteration = epoch * len(train_dataloader) + batch_idx

            if iteration % config["print_every_n"] == (config["print_every_n"] - 1):
                train_loss = train_loss_running / config["print_every_n"]
                print(f"[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss:.6f}")

                writer.add_scalar(
                    "Training loss",
                    train_loss,
                    iteration,
                )

                train_loss_running = 0.0

            if iteration % config["validate_every_n"] == (
                config["validate_every_n"] - 1
            ):
                model.eval()
                loss_val = 0.0
                for batch_idx_val, batch_val in enumerate(val_dataloader):
                    # Move batch to device
                    ShapeNetDataset.move_batch_to_device(batch_val, config["device"])
                    input_images = batch["Image"]
                    input_voxels = batch["3D_prior"]
                    target_voxels = batch["GT"]

                    with torch.no_grad():
                        for _ in range(config["iter"]):
                            input_voxels = model(input_voxels, input_images)
                        pred_val = input_voxels.squeeze(1)
                        loss_val += loss_criterion(pred_val, target_voxels)

                loss_val /= len(val_dataloader)

                #early stop
                if loss_val > last_loss:
                    trigger_times += 1

                    if trigger_times >= patience:
                        print('Early stopping!')
                        return model

                else:
                    trigger_times = 0

                last_loss = loss_val

                if loss_val < best_loss_val:
                    torch.save(
                        model.state_dict(),
                        f'runs/{dt_string}-{config["experiment_name"]}/{dt_string}-{config["experiment_name"]}.ckpt',
                    )
                    best_loss_val = loss_val
                print(
                    f"[{epoch:03d}/{batch_idx:05d}] loss_val: {loss_val:.6f} | best_loss_val: {best_loss_val:.6f}"
                )
                writer.add_scalar(
                    "Validation loss",
                    loss_val,
                    iteration,
                )

                model.train()
    return model


def test(
    model: Model,
    test_dataloader: DataLoader,
    config: Dict,
):
    """
    Test the model using unseen shape.

    Args:
        model (Model): The trained model.
        test_dataloader_shape (DataLoader): Provide test data whose shape was not seen by the model.

    Retruns:
        tbd
    """

    loss_criterion = torch.nn.BCELoss()
    loss_criterion.to(config["device"])

    model.eval()

    loss_test = 0.0
    iou_test = 0.0

    with torch.no_grad():

        # test_shape
        for batch_idx, batch in enumerate(test_dataloader):
            # Move batch to device
            ShapeNetDataset.move_batch_to_device(batch, config["device"])
            input_images = batch["Image"]
            input_voxels = batch["3D_prior"]
            target_voxels = batch["GT"]

            # Perform forward pass
            for _ in range(config["iter"]):
                input_voxels = model(input_voxels, input_images)
            pred_test = input_voxels.squeeze(1)

            loss_test_running = loss_criterion(pred_test, target_voxels)
            loss_test += loss_test_running

            # Compute IoU
            iou_running = iou(target_voxels, pred_test, threshold=0.4)
            iou_test += iou_running

        loss_test /= len(test_dataloader)
        iou_test /= len(test_dataloader)

        print(f"loss_test: {loss_test:.6f} | IoU_test: {iou_test:.6f}")


def iou(true_voxels, pred_voxels, threshold=0.4):
    bool_true_voxels = true_voxels > threshold
    bool_pred_voxels = pred_voxels > threshold
    total_union = (bool_true_voxels | bool_pred_voxels).sum()
    total_intersection = (bool_true_voxels & bool_pred_voxels).sum()
    return total_intersection / total_union


def main(model, config):
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
    train_dataset = ShapeNetDataset("train", config["prior_k"])
    train_dataloader = DataLoader(
        train_dataset,  # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config["batch_size"],  # The size of batches is defined here
        shuffle=True,
        num_workers=4,
    )

    val_dataset = ShapeNetDataset("shape_val", config["prior_k"])
    val_dataloader = DataLoader(
        val_dataset,  # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config["batch_size"],  # The size of batches is defined here
        shuffle=True,
        num_workers=4,
    )

    test_dataset = ShapeNetDataset("shape_test", config["prior_k"])
    test_dataloader = DataLoader(
        test_dataset,  # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config["batch_size"],  # The size of batches is defined here
        shuffle=True,
        num_workers=4,
    )

    # Instantiate model
    model = Model()

    # Move model to specified device
    model.to(config["device"])

    # Start training
    train(
        model,
        train_dataloader,
        val_dataloader,
        config,
    )

    # Start testing
    test(
        model,
        test_dataloader,
        config,
    )

    return model
