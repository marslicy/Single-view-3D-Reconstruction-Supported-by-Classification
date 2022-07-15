from pathlib import Path

import numpy as np
from tqdm import tqdm

from split.binvox_rw import read_as_3d_array

voxroot = "/home/yang/Codes/ml3d_final_dataset/ShapeNetVox32"
# load the shapes in dictionary, class as the key, all shapes as a list

file = "split/train.txt"
train = Path(file).read_text().splitlines()

shape_dict = {}  # if the category not exists, create key and empty list, add the

for line in train:
    category, shape = line.split("/")
    if str(category) not in shape_dict.keys():
        shape_dict[str(category)] = []
        shape_dict[str(category)].append(shape)
    else:
        shape_dict[str(category)].append(shape)


def get_shape_voxels(category_id, shape_id, vox_path=voxroot):
    """
    Utility method for reading a ShapeNet voxel grid from disk, reads voxels from binvox format on disk as 3d numpy arrays
    :param shapenet_id: Shape ID of the form <shape_class>/<shape_identifier>, e.g. 03001627/f913501826c588e89753496ba23f2183
    :return: a numpy array representing the shape voxels
    """
    with open(f"{vox_path}/{category_id}/{shape_id}/model.binvox", "rb") as fptr:
        voxel = read_as_3d_array(fptr).astype(np.float32)
    return voxel


# Compute all the priors

save_dir = "data/prior"


for key in shape_dict.keys():
    all_shapes = np.empty(
        [1, 32, 32, 32]
    )  # initialize the full_shape array, first entry will be deleted later
    for shapeid in tqdm(shape_dict[key]):
        # get every shape
        np_shape = np.array(get_shape_voxels(key, str(shapeid)))
        np_shape = np_shape[None, :, :, :]
        # stack shapes into [n,32,32,32]
        all_shapes = np.concatenate((all_shapes, np_shape), axis=0)

    # delete the initialization
    all_shapes = all_shapes[1:]
    # get mean from it
    full_prior = np.mean(all_shapes, axis=0)
    # save it with the name of category_id at the prior dir
    np.save(f"{save_dir}/{key}.npy", full_prior)
