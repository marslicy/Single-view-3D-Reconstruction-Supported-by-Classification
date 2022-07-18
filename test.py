# %%
import torch
from torch.utils.data import DataLoader

from data.shapenet import ShapeNetDataset
from model.model import Model

dataset = ShapeNetDataset("train")
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
model = Model()
for batch in data_loader:
    pred = model(batch["3D_prior"], batch["Image"])
    print(pred.shape)
    break
# %%   
from model.train import train
from model.train import main
config = {
    "experiment_name": "test",
    "device": "cpu",  # or 'cuda:0'
    "batch_size": 10,
    "resume_ckpt": None,
    "learning_rate": 0.0001,
    "max_epochs": 1,
    "print_every_n": 5,
    "validate_every_n":10,
}
main(model, config)

# %%
