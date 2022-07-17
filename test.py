# %%
import torch
from torch.utils.data import DataLoader

from data.shapenet import ShapeNetDataset
from model.model import Model

dataset = ShapeNetDataset("train", prior_k=5)
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
model = Model()
for batch in data_loader:
    pred = model(batch["3D_prior"], batch["Image"])
    print(pred.shape)
    break
# %%
