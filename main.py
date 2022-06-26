import torch

from model.train import main

if __name__ == "__main__":
    """
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

    config = {}
    device = torch.device("cpu")
    if torch.cuda.is_available() and config["device"].startswith("cuda"):
        device = torch.device(config["device"])
        print("Using device:", config["device"])
    else:
        print("Using CPU")
    main(config)
