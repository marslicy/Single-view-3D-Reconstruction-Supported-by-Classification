import numpy as np

from configs.path import voxroot
from split.binvox_rw import read_as_3d_array

# read overfit list

with open("split/overfit_shape.txt") as file:
    # file_list = file.read().splitlines()
    lines = []
    for line in file:
        lines.append(line.rstrip())

# print(len(lines))

# read shape files

test = lines[5]


def get_shape_voxels(shapenetid):
    category_id, shape_id = shapenetid.split("/")
    with open(f"{voxroot}/{category_id}/{shape_id}/model.binvox", "rb") as fptr:
        voxel = read_as_3d_array(fptr).astype(np.float32)
    return voxel


voxel = get_shape_voxels(test)
