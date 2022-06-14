import json
from pathlib import Path

import numpy as np
import torch

from configs.path import imgroot, voxroot
from split.binvox_rw import read_as_3d_array


class ShapeNetDataset(torch.utils.data.Dataset):
    """
    Dataset for loading ShapeNet Voxels from disk
    """

    num_classes = 13  # currently we have 13 classes
    vox_path = voxroot
    img_path = imgroot
    class_name_mapping = json.loads(Path("split/shape_info.json").read_text())
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split, val_view=2, test_view=2):
        """
        :param split: one of 'train', 'view_val', 'shape_val', 'view_test', 'test_shape' - for training, validation or overfitting split
        """
        super().__init__()
        assert split in [
            "train",
            "view_val",
            "shape_val",
            "view_test",
            "shape_test",
        ]

        # keep track of shapes based on split
        self.items = Path(f"split/{split}.txt").read_text().splitlines()

        self.split = split  # split can be used in other places
        # TODO define the number of image in  view_val and view_test. The overall image of a shape is 24, therefore the remaining part will be assigned to train
        self.val_view = val_view
        self.test_view = test_view

    def __getitem__(self, index):
        """
        we implement the following splits for the dataset: train, view_val, shape_val, overfit_shape, view
        :param index: index of the dataset sample that will be returned
        :return: a dictionary of data corresponding to the shape. In particular, this dictionary has keys
                 "name", given as "<shape_category>/<shape_identifier>",
                 "voxel", a 1x32x32x32 numpy float32 array representing the shape
                 "label", a number in [0, 12] representing the class of the shape
        """
        item = self.items[index]
        item_class = item.split("/")[0]
        # read voxels from binvox format on disk as 3d numpy arrays
        voxels = self.get_shape_voxels(item)

        return {
            "class": np.random((1, 3, 128, 128)),
            # we add an extra dimension as the channel axis, since pytorch 3d tensors are Batch x Channel x Depth x Height x Width
            "encoder": np.random((1, 3, 128, 128)),
            "GT": ShapeNetDataset.classes.index(item_class),
            "3D": voxels[np.newaxis, :, :, :],
            # label is 0 indexed position in sorted class list, e.g. 02691156 is label 0, 02828884 is label 1 and so on.
            "ID": item,
        }

    def __len__(self):
        """
        :return: length of the dataset
        """
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch["voxel"] = batch["voxel"].to(device)
        batch["label"] = batch["label"].to(device)

    @staticmethod
    def get_shape_voxels(shapenetid):
        """
        Utility method for reading a ShapeNet voxel grid from disk, reads voxels from binvox format on disk as 3d numpy arrays
        :param shapenet_id: Shape ID of the form <shape_class>/<shape_identifier>, e.g. 03001627/f913501826c588e89753496ba23f2183
        :return: a numpy array representing the shape voxels
        """
        category_id, shape_id = shapenetid.split("/")
        with open(
            f"{ShapeNetDataset.vox_path}/{category_id}/{shape_id}/model.binvox", "rb"
        ) as fptr:
            voxel = read_as_3d_array(fptr).astype(np.float32)
        return voxel

    @staticmethod
    def get_image_data(imageid):
        pass
