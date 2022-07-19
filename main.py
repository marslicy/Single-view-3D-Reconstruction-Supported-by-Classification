import torch

from model.model import Model
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
        "experiment_name": "full_prior_1_iter",
        "device": "cpu",  # or 'cuda:0'
        "batch_size": 128,
        "prior_k": "full",
        "iter": 1,
        "resume_ckpt": None,
        "learning_rate": 0.1,
        "max_epochs": 100,
        "print_every_n": 100,
        "validate_every_n": 200,
    }
    if torch.cuda.is_available():
        config["device"] = "cuda:0"
    print("Using device:", config["device"])

    print("full_prior_1_iter")
    model = Model()
    model = main(model, config)

    print("empty_prior_1_iter")
    config["prior_k"] = 0
    config["experiment_name"] = "empty_prior_1_iter"
    model = Model()
    model = main(model, config)

    print("1_prior_1_iter")
    config["prior_k"] = 1
    config["experiment_name"] = "1_prior_1_iter"
    model = Model()
    model = main(model, config)

    print("1_prior_2_iter")
    config["iter"] = 2
    config["prior_k"] = 1
    config["experiment_name"] = "1_prior_2_iter"
    model = Model()
    model = main(model, config)
