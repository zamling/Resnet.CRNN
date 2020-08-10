
import torch
from torch import randperm
import bisect
from torch._utils import _accumulate
from torch.utils.data import Dataset
import random

class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        return self.cumulative_sizes



class ConcatDataset_V2(Dataset):
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = 90698 #len(ReCTS)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset_V2, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.len_MJ = 8917543
        self.len_Synth = 7266164

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        '''
        0th data : ReCTS
        1st data : NIPS 2014
        2nd data : CVPR 2016
        '''
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        elif dataset_idx == 1:
            #the dataset is NIPS 2014
            sample_idx = random.randint(0,self.len_MJ - 1)
        elif dataset_idx == 2:
            sample_idx = random.randint(0,self.len_Synth - 1)
        else:
            raise RuntimeError
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        return self.cumulative_sizes