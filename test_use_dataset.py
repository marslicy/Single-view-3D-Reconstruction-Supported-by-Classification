# %%

from data.shapenet import ShapeNetDataset
from data.shapenet_loader import ShapeNetDataLoader

split = "view_val"
dataset = ShapeNetDataset(split, val_view=3, test_view=3)

item1 = dataset[0]
item2 = dataset[1]
# %%
# %%

trainloader = ShapeNetDataLoader(dataset, batch_size=3, shape_num=3)

i = 0
for batch in trainloader:
    i += 1
    import pdb

    pdb.set_trace()
    print(type(batch))
# %%
