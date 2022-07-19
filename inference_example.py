from torch.utils.data import DataLoader

from data.shapenet import ShapeNetDataset
from model.inference import Inference

split = "new_categories"
device = "cuda:0"

dataset = ShapeNetDataset(split)

loader = DataLoader(dataset, batch_size=8)

ckpt = "ckpts/07-19-20-23-1_prior_1_iter.ckpt"

inference = Inference(device=device, ckpt=ckpt)

for i, batch in enumerate(loader):
    result = inference.reconstruct(batch["Image"].to(device))

    if i > 5:
        break


print("successfully inferenced!")
