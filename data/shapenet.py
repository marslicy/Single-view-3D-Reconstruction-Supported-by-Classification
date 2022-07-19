import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from configs.path import imgroot, priorroot, voxroot
from split.binvox_rw import read_as_3d_array


class ShapeNetDataset(torch.utils.data.Dataset):
    """
    Dataset for loading ShapeNet Voxels from disk
    """

    num_classes = 13  # currently we have 13 classes
    vox_path = voxroot
    img_path = imgroot
    prior_path = priorroot
    class_name_mapping = json.loads(Path("split/shape_info.json").read_text())
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split, prior_k=10):
        """
        :param split: one of 'train', 'shape_test', 'shape_val', "new_categories"
        """
        super().__init__()
        assert split in [
            "train",
            "shape_val",
            "shape_test",
            "new_categories",
        ]

        self.items = Path(f"./split/{split}.txt").read_text().splitlines()

        self.split = split  # split can be used in other places

        self.split_dict = self.load_shape_as_dict()

        # determine the prior type
        self.prior_k = prior_k
        if isinstance(self.prior_k, int):
            if self.prior_k > 0:
                self.get_prior = self.get_k_prior_by_category
            elif self.prior_k == 0:
                self.get_prior = self.get_empty_prior
        else:
            self.get_prior = self.get_full_prior

    def __getitem__(self, index):
        """
        we implement the following splits for the dataset: train, view_val, shape_val, overfit_shape, view
        :param index: index of the dataset sample that will be returned
        :return: a dictionary of data corresponding to the shape. In particular, this dictionary has keys
                 "3D_prior", a 32x32x32 numpy ndarray
                 "GT", a 32x32x32 numpy ndarray
                 "Image", 3x127x127 numpy ndarray
                 "ID", string that in form "category_id/shape_id"
        """
        item = self.items[index]
        category = item.split("/")[0]
        # read voxels from binvox format on disk as 3d numpy arrays
        voxel = torch.from_numpy(self.get_shape_voxels(item)).float()

        prior = torch.from_numpy(self.get_prior(category)).float()

        image = torch.from_numpy(self.get_image_data(item)).float()

        return {
            "3D_prior": prior,
            "Image": image,
            "GT": voxel,
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
        batch["3D_prior"] = batch["3D_prior"].to(device)
        batch["Image"] = batch["Image"].to(device)
        batch["GT"] = batch["GT"].to(device)

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

        assert self.split in ["train", "shape_val", "shape_test", "new_categories"]

        idx = np.random.randint(0, 24)

        img_idx = str(idx).zfill(2)
        category_id, shape_id = shapenetid.split("/")
        path = f"{imgroot}/{category_id}/{shape_id}/rendering/{img_idx}.png"

        img = cv2.imread(path)
        img = cv2.resize(img, (127, 127))
        img = np.transpose(img, (2, 0, 1))

        return img

    def get_k_prior_by_category(self, category):
        """
        Randomly selected k shapes in the category and calculate the average shape

        Args:
            item (string): must be an integer in the range [0, num_shape)
            number (int): must be a positive integer

        Returns:
            k_prior(np.ndarray): averange shape in shape (1, 32, 32, 32)
        """

        # sample from the given category
        all = self.split_dict[category]
        sampled = random.sample(all, self.prior_k)

        # initialize the full_shape array, first entry will be deleted later
        k_shapes = np.empty([1, 32, 32, 32])
        for shapeid in sampled:
            # get every shape
            np_shape = np.array(self.get_shape_voxels(f"{category}/{shapeid}"))
            np_shape = np_shape[None, :, :, :]
            # stack shapes into [n,32,32,32]
            k_shapes = np.concatenate((k_shapes, np_shape), axis=0)

        # delete the initialization
        all_shapes = k_shapes[1:]
        # assert all_shapes.shape == (self.prior_k, 32, 32, 32)
        # get mean from it
        k_prior = np.expand_dims(np.mean(all_shapes, axis=0), axis=0)
        # assert k_prior.shape == (1, 32, 32, 32)

        return k_prior

    def get_full_prior(self, category):
        """
        Read the full prior computed in advanced from the whole training set.
        Arg
            category(string): the category

        Returns
            full_prior(np.ndarray): the shape prior we want by [1, 32,32,32]
        """

        prior_dir = "data/prior"

        full_prior = np.expand_dims(np.load(f"{prior_dir}/{category}.npy"), axis=0)

        return full_prior

    def get_empty_prior(self, category):
        """
        Read the full prior computed in advanced from the whole training set.
        Arg
            category(string): the category, not used, just for keep format

        Returns
            empty_prior(np.ndarray): the shape prior by [1, 32,32,32] with all zero
        """

        empty_prior = np.zeros((1, 32, 32, 32))

        return empty_prior

    def load_shape_as_dict(self):
        """
        Read the split as dictionary in classes, help to simplify the get_k_prior_by_category process

        Arg
            self.items(list): items

        Returns
            split_dict(dictionary): the dictionary
        """
        split_dict = {}
        for item in self.items:
            category_id, shape_id = item.split("/")
            if category_id not in split_dict.keys():
                split_dict[category_id] = [shape_id]
            else:
                split_dict[category_id].append(shape_id)

        return split_dict
