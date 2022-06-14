"""Definition of Dataloader"""

import numpy as np

# TODO define different behavior for test and val split
# More specifically, we should 1) use the same image for both encoder 2) the shape_num should be set to 1 (dataloader) 3) all views in val/test should be used (no random pick rather a

# In order to achive 3), a method called load_image_by_idx should be implemented at Dataset, if split is not train, load enc and cls with this method. Dataloader should also be informed with the num of val and test imgs per shape, so that it will change the shape_num to this number. This way it will pick all the imgs belong to this split.


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """

    def __init__(
        self, dataset, batch_size=1, shape_num=3, shuffle=False, drop_last=False
    ):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.shape_num = shape_num
        # use this split to set shapenum
        self.split = self.dataset.split
        if self.split != "train":
            self.shape_num = 1

    def __iter__(self):
        def combine_batch_dicts(batch):
            """
            Combines a given batch (list of dicts) to a dict of numpy arrays
            :param batch: batch, list of dicts
                e.g. [{k1: v1, k2: v2, ...}, {k1:, v3, k2: v4, ...}, ...]
            :returns: dict of numpy arrays
                e.g. {k1: [v1, v3, ...], k2: [v2, v4, ...], ...}
            """
            batch_dict = {}
            for data_dict in batch:
                for key, value in data_dict.items():
                    if key not in batch_dict:
                        batch_dict[key] = []
                    batch_dict[key].append(value)
            return batch_dict

        def batch_to_numpy(batch):
            """Transform all values of the given batch dict to numpy arrays"""
            numpy_batch = {}
            for key, value in batch.items():
                numpy_batch[key] = np.array(value)
            return numpy_batch

        if self.shuffle:
            index_iterator = iter(np.random.permutation(len(self.dataset)))
        else:
            index_iterator = iter(range(len(self.dataset)))

        batch = []
        terminate_num = 0
        for index in index_iterator:
            for i in range(self.shape_num - terminate_num):
                batch.append(self.dataset[index])
                # TODO modify here to sample one index multiple times,
                # should leave a mark about how many times (marked as terminate_num) the last item has been used,
                # so that we will start the next batch with (shape_num - n) times the index
                if len(batch) == self.batch_size:
                    terminate_num = i
                    yield batch_to_numpy(combine_batch_dicts(batch))
                    batch = []
            terminate_num = 0

        if len(batch) > 0 and not self.drop_last:
            yield batch_to_numpy(combine_batch_dicts(batch))

    def __len__(self):
        length = None

        if self.drop_last:
            length = len(self.dataset) // self.batch_size
        else:
            length = int(np.ceil(len(self.dataset) / self.batch_size))

        return length
