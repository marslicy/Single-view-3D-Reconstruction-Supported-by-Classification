from typing import Dict

from model import Model
from torch.utils.data import DataLoader


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
                       'experiment_name': name of the experiment, checkpoint will be saved to folder "runs/<experiment_name>"
                       'device': device on which model is trained, e.g. 'cpu' or 'cuda:0'
                       'batch_size': batch size for training and validation dataloaders
                       'resume_ckpt': None if training from scratch, otherwise path to checkpoint (saved weights)
                       'learning_rate': learning rate for optimizer
                       'max_epochs': total number of epochs after which training should stop
                       'print_every_n': print train loss every n iterations
                       'validate_every_n': print validation loss and validation accuracy every n iterations
    """
    pass


def test(
    model: Model, test_dataloader_view: DataLoader, test_dataloader_shape: DataLoader
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
    pass
