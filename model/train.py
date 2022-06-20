from typing import Dict
from model.model import Model
from data.shapenet import ShapeNetDataset
from data.shapenet_loader import ShapeNetDataLoader
import torch
from pathlib import Path
from torch.utils.data import DataLoader


def train(
    model: Model,
    train_dataloader: DataLoader,
    val_dataloader_view: DataLoader,
    val_dataloader_shape: DataLoader,
    device,
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
    loss_criterion_class = torch.nn.CrossEntropyLoss()
    loss_criterion_class.to(device)
    loss_criterion_3d = torch.nn.L1Loss()
    loss_criterion_3d.to(device)

    optimizer = torch.optim.Adam(model.parameters(), config['learning_rate'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5) # not necessary

    model.train()

    train_loss_running = 0.
    best_loss = float('inf')
    val_view_loss_running = 0.0
    val_shape_loss_running = 0.0

    for epoch in range(config['max_epochs']):

        for batch_idx, batch in enumerate(train_dataloader): 
            # Move batch to device
            ShapeNetDataset.move_batch_to_device(batch, device)
            x1 = batch['class']
            x2 = batch['encoder']
            y1 = batch['GT']
            y2 = batch['3D']

            optimizer.zero_grad()

            # Perform forward pass
            pred_class, pred_3d = model(x1, x2)

            loss_class = loss_criterion_class(pred_class, y1)
            loss_3d = loss_criterion_3d(pred_3d, y2)
            loss = loss_class + loss_3d
        
            # Backward
            loss.backward()
            # Update network parameters
            optimizer.step()
            # loss logging
            train_loss_running += loss.item()
            iteration = epoch * len(train_dataloader) + batch_idx

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                train_loss = train_loss_running / config["print_every_n"]
                print(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss:.6f}')

                # save best train model and latent codes
                if train_loss < best_loss:
                    torch.save(model.state_dict(), f'3dml-project/runs/{config["experiment_name"]}/model_best.ckpt')
                    # torch.save(latent_vectors.state_dict(), f'exercise_3/runs/{config["experiment_name"]}/latent_best.ckpt')
                    best_loss = train_loss

                train_loss_running = 0.
        
        # val_view
        for batch_idx, batch in enumerate(val_dataloader_view): 
            # Move batch to device
            ShapeNetDataset.move_batch_to_device(batch, device)
            x1 = batch['class']
            x2 = batch['encoder']
            y1 = batch['GT']
            y2 = batch['3D']

            optimizer.zero_grad()

            # Perform forward pass
            pred_class, pred_3d = model(x1, x2)

            loss_class = loss_criterion_class(pred_class, y1)
            loss_3d = loss_criterion_3d(pred_3d, y2)
            loss = loss_class + loss_3d
        
            # Backward
            loss.backward()
            # Update network parameters
            optimizer.step()
            # loss logging
            val_view_loss_running += loss.item()
            iteration = epoch * len(val_dataloader_view) + batch_idx

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                val_view_loss = val_view_loss_running / config["print_every_n"]
                print(f'[{epoch:03d}/{batch_idx:05d}] val_view_loss: {val_view_loss:.6f}')

                val_view_loss_running = 0.

        # val_shape 
        for batch_idx, batch in enumerate(val_dataloader_shape): 
            # Move batch to device
            ShapeNetDataset.move_batch_to_device(batch, device)
            x1 = batch['class']
            x2 = batch['encoder']
            y1 = batch['GT']
            y2 = batch['3D']

            optimizer.zero_grad()

            # Perform forward pass
            pred_class, pred_3d = model(x1, x2)

            loss_class = loss_criterion_class(pred_class, y1)
            loss_3d = loss_criterion_3d(pred_3d, y2)
            loss = loss_class + loss_3d
        
            # Backward
            loss.backward()
            # Update network parameters
            optimizer.step()
            # loss logging
            val_shape_loss_running += loss.item()
            iteration = epoch * len(val_dataloader_shape) + batch_idx

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                val_shape_loss = val_shape_loss_running / config["print_every_n"]
                print(f'[{epoch:03d}/{batch_idx:05d}] val_shape_loss: {val_shape_loss:.6f}')

                val_shape_loss_running = 0.


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

    # declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # create dataloaders
    train_dataset = ShapeNetDataset('train', config['val_view'], config['test_view'])
    train_dataloader = ShapeNetDataLoader(
        train_dataset,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
    )

    val_dataset_view = ShapeNetDataset('view_val', config['val_view'], config['test_view'])
    val_dataloader_view = ShapeNetDataLoader(
        val_dataset_view,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
    )

    val_dataset_shape = ShapeNetDataset('shape_val', config['val_view'], config['test_view'])
    val_dataloader_shape = ShapeNetDataLoader(
        val_dataset_shape,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
    )

    # Instantiate model
    model = Model()

    # Move model to specified device
    model.to(device)

    # Create folder for saving checkpoints
    Path(f'3dml-project/runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Start training
    train(model, train_dataloader, val_dataloader_view, val_dataloader_shape, device, config)   


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
















