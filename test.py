# %%
# test dataset
from data.shapenet import ShapeNetDataset
from data.shapenet_loader import ShapeNetDataLoader

split = "view_val"
dataset = ShapeNetDataset(split, val_view=3, test_view=3)

item1 = dataset[0]
item2 = dataset[1]

# %%
# test trainloader
trainloader = ShapeNetDataLoader(dataset, batch_size=3, shape_num=3)

i = 0
for batch in trainloader:
    i += 1
    import pdb

    pdb.set_trace()
    print(type(batch))

# %%
# test model
import torch

from model.model import Model

inputs = torch.rand((5, 3, 127, 127))
output_class, output_3d = Model()(inputs, inputs)
assert output_class.shape == (5, 13)
assert output_3d.shape == (5, 32, 32, 32)

# %%
