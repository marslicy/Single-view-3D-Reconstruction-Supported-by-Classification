import torch

from data.shapenet import ShapeNetDataset
from data.shapenet_loader import ShapeNetDataLoader
from model.model import Model
from model.train import main, test

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
        "experiment_name": "a1b1",
        "device": "cuda:0",  # or 'cuda:0 cpu'
        "batch_size": 64,
        "resume_ckpt": None,
        "learning_rate": 0.001,
        "max_epochs": 150,
        "print_every_n": 100,
        "validate_every_n": 200,
        "val_view": 1,
        "test_view": 1,
        "shape_num": 1,
        "a": 1,
        "b": 1,
        "global_feature_size": 128,
        "local_feature_size": 128,
        "num_class": 13,
    }
    if torch.cuda.is_available():
        config["device"] = "cuda"
    print("Using device:", config["device"])

    # for training
    config["a"] = 0.5
    config["b"] = 1
    config["experiment_name"] = "a0.5b1"

    model = Model(
        config["global_feature_size"], config["local_feature_size"], config["num_class"]
    )
    model = main(model, config)

    config["a"] = 0.1
    config["b"] = 1
    config["experiment_name"] = "a0.1b1"

    model = Model(
        config["global_feature_size"], config["local_feature_size"], config["num_class"]
    )
    model = main(model, config)

    config["a"] = 1
    config["b"] = 1
    config["experiment_name"] = "a1b1"

    model = Model(
        config["global_feature_size"], config["local_feature_size"], config["num_class"]
    )
    model = main(model, config)
    # end training

    # for test
    test_dataset_view = ShapeNetDataset(
        "view_test", config["val_view"], config["test_view"]
    )
    test_dataloader_view = ShapeNetDataLoader(
        test_dataset_view,  # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config["batch_size"],  # The size of batches is defined here
        shuffle=True,
    )

    test_dataset_shape = ShapeNetDataset(
        "shape_test", config["val_view"], config["test_view"]
    )
    test_dataloader_shape = ShapeNetDataLoader(
        test_dataset_shape,  # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config["batch_size"],  # The size of batches is defined here
        shuffle=True,
    )

    model = Model(
        config["global_feature_size"], config["local_feature_size"], config["num_class"]
    )
    model.load_state_dict(torch.load("./runs/a0.5b1/val_shape_model_best.ckpt"))
    model = model.to(config["device"])

    config["a"] = 0.5
    config["b"] = 1

    test(
        model,
        test_dataloader_view,
        test_dataloader_shape,
        config,
    )
    # end test
