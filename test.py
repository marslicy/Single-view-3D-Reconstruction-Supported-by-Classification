# %%
# import here
import torch
from cv2 import DRAW_MATCHES_FLAGS_DEFAULT

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
# %%
