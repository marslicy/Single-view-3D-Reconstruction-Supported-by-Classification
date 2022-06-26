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

    config = {
        "experiment_name": "train_result",
        "device": "cpu",  # or 'cuda:0'
        "batch_size": 10,
        "resume_ckpt": None,
        "learning_rate": 0.0001,
        "max_epochs": 1,
        "print_every_n": 1,
        "validate_every_n": 1,
        "val_view": 3,
        "test_view": 3,
        "a": 1,
        "b": 1,
        "global_feature_size": 128,
        "local_feature_size": 128,
        "num_class": 13,
    }
    if torch.cuda.is_available():
        config["device"] = "cuda"
    print("Using device:", config["device"])
    main(config)
