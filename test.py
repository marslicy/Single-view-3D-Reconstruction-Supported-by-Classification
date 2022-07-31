# %%
# import here
import torch

from data.shapenet import ShapeNetDataset
from data.shapenet_loader import ShapeNetDataLoader
from model import train
from model.model import Model
from util.visualization import visualize_2d, visualize_3d

# %%
# test trainloader
split = "train"
dataset = ShapeNetDataset(split, val_view=3, test_view=3)
print(len(dataset))
trainloader = ShapeNetDataLoader(dataset, batch_size=3, shape_num=3)

# i = 0
# for batch in trainloader:
#     i += 1
#     # import pdb
#     print(batch.keys())
#     # pdb.set_trace()


# %%
# test model
inputs = torch.rand((5, 3, 127, 127))
output_class, output_3d = Model()(inputs, inputs)
assert output_class.shape == (5, 13)
assert output_3d.shape == (5, 32, 32, 32)


# %%
# test train
config = {
    "experiment_name": "test",
    "device": "cpu",  # or 'cuda:0'
    "batch_size": 10,
    "resume_ckpt": None,
    "learning_rate": 0.001,
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

train.main(config)

# %%
# test visualization
split = "view_val"
dataset = ShapeNetDataset(split, val_view=3, test_view=3)
trainloader = ShapeNetDataLoader(dataset, batch_size=3, shape_num=3)
batch = trainloader.__iter__().__next__()
visualize_3d(batch["3D"][0], batch["3D"][0], "test")
visualize_2d(batch["class"][0])
# %%
# test test
config = {
    "experiment_name": "test",
    "device": "cpu",  # or 'cuda:0'
    "batch_size": 10,
    "resume_ckpt": None,
    "learning_rate": 0.001,
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
test_dataset_view = ShapeNetDataset(
    "view_test", config["val_view"], config["test_view"]
)
test_dataloader_view = ShapeNetDataLoader(
    test_dataset_view[
        :10
    ],  # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
    batch_size=config["batch_size"],  # The size of batches is defined here
)

test_dataset_shape = ShapeNetDataset(
    "shape_test", config["val_view"], config["test_view"]
)
test_dataloader_shape = ShapeNetDataLoader(
    test_dataset_shape[
        :10
    ],  # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
    batch_size=config["batch_size"],  # The size of batches is defined here
)
model = Model(
    config["global_feature_size"], config["local_feature_size"], config["num_class"]
)
model.load_state_dict(
    torch.load(
        f'./runs/{config["experiment_name"]}/model_best.ckpt',
        map_location="cpu",
    )
)
train.test(
    model,
    test_dataloader_view,
    test_dataloader_shape,
    config,
)
import torch

# %%
from data.shapenet import ShapeNetDataset
from data.shapenet_loader import ShapeNetDataLoader

config = {
    "experiment_name": "a1b1",
    "device": "cpu",  # or 'cuda:0 cpu'
    "batch_size": 128,
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
for batch_idx, batch in enumerate(test_dataloader_shape):
    # Move batch to device
    ShapeNetDataset.move_batch_to_device(batch, config["device"])
    x1 = batch["class"]
    x2 = batch["encoder"]
    y1 = batch["GT"]
    y2 = batch["3D"]

    print(torch.eq(x1, x2))
    break

# %%
val_dataset_shape = ShapeNetDataset("shape_val", 1, 1)
val_dataloader_shape = ShapeNetDataLoader(
    val_dataset_shape,  # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
    batch_size=12,  # The size of batches is defined here
    shuffle=True,
)
for batch in val_dataloader_shape:
    x1 = batch["class"]
    x2 = batch["encoder"]
    y1 = batch["GT"]
    y2 = batch["3D"]
    print(torch.eq(x1, x2).all())
# %%
