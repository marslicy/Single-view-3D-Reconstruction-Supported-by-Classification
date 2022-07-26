import json
from pathlib import Path

import cv2
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
        :param split: one of 'train', 'view_val', 'shape_val', 'view_test', 'shape_test' - for training, validation or overfitting split
        """
        super().__init__()
        assert split in [
            "train",
            "view_val",
            "shape_val",
            "view_test",
            "shape_test",
        ]

        # keep track of shapes based on split, while view_val and view_test uses train split
        if split in ["train", "shape_val", "shape_test"]:
            self.items = Path(f"split/{split}.txt").read_text().splitlines()
        else:
            self.items = Path("split/train.txt").read_text().splitlines()
            if split == "view_test":
                self.img_counter = ShapeNetDataset.idx_generator(test_view)
            if split == "view_val":
                self.img_counter = ShapeNetDataset.idx_generator(val_view)

        self.split = split  # split can be used in other places
        # TODO define the number of image in  view_val and view_test. The overall image of a shape is 24, therefore the remaining part will be assigned to train
        self.val_view = val_view
        self.test_view = test_view
        self.train_num = 24 - val_view - test_view

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

        # implement the different geting logic here

        if self.split in ["train"]:
            enc_img = self.get_image_data(item)
            cls_img = self.get_image_data(item)
        elif self.split in ["shape_val", "shape_test"]:
            enc_img = self.get_image_data(item)
            cls_img = enc_img
        elif self.split in ["view_val", "view_test"]:
            idx = next(self.img_counter)
            enc_img = self.get_image_by_idx(item, idx)
            cls_img = enc_img

        return {
            "class": cls_img,
            "encoder": enc_img,
            "GT": ShapeNetDataset.classes.index(item_class),
            "3D": voxels,
            "ID": item,
        }

    @staticmethod
    def idx_generator(length):
        i = 0
        while True:
            yield i % length
            i += 1

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
        batch["class"] = batch["class"].to(device)
        batch["encoder"] = batch["encoder"].to(device)
        batch["GT"] = batch["GT"].to(device)
        batch["3D"] = batch["3D"].to(device)

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

    def get_image_data(self, shapenetid):

        assert self.split in ["train", "shape_val", "shape_test"]
        if self.split == "train":
            idx = np.random.randint(0, self.train_num)
        else:
            idx = np.random.randint(0, 24)

        img_idx = str(idx).zfill(2)
        category_id, shape_id = shapenetid.split("/")
        path = f"{imgroot}/{category_id}/{shape_id}/rendering/{img_idx}.png"

        img = cv2.imread(path)
        img = cv2.resize(img, (127, 127))
        img = np.transpose(img, (2, 0, 1))

        return img

    def get_image_by_idx(self, shapenetid, idx):
        # the index should be [0, split_length]

        assert self.split in ["view_val", "view_test"]
        if self.split == "view_val":
            # index of view_val should be in [train_end , train_end + val_length]
            img_idx = idx + self.train_num
        else:
            # index of view_val should be in [self.train_num + self.val_view , 24]
            img_idx = idx + self.train_num + self.val_view

        img_idx = str(idx).zfill(2)
        category_id, shape_id = shapenetid.split("/")
        path = f"{imgroot}/{category_id}/{shape_id}/rendering/{img_idx}.png"
        img = cv2.imread(path)
        img = cv2.resize(img, (127, 127))
        img = np.transpose(img, (2, 0, 1))
        return img
