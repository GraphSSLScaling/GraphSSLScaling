import torch
import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import f1_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV


def get_split(num_samples: int, train_ratio: float = 0.8, test_ratio: float = 0.1, downstream_ratio = 0.1, dataset = 'Cora'):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    downstream_size = int(downstream_ratio*train_size)
    indices = torch.load("./split/{}.pt".format(dataset))
    print(f"{dataset} downstram_train:{downstream_size} test_size:{num_samples-train_size-test_size}")
    return {
        'train': indices[:downstream_size],
        'valid':indices[train_size:train_size+test_size],
        'test': indices[train_size+test_size:]
    } 


def from_predefined_split(data):
    assert all([mask is not None for mask in [data.train_mask, data.test_mask, data.val_mask]])
    num_samples = data.num_nodes
    indices = torch.arange(num_samples)
    return {
        'train': indices[data.train_mask],
        'valid': indices[data.val_mask],
        'test': indices[data.test_mask]
    }


def split_to_numpy(x, y, split):
    keys = ['train', 'test', 'valid']
    objs = [x, y]
    return [obj[split[key]].detach().cpu().numpy() for obj in objs for key in keys]


def get_predefined_split(x_train, x_val, y_train, y_val, return_array=True):
    test_fold = np.concatenate([-np.ones_like(y_train), np.zeros_like(y_val)])
    ps = PredefinedSplit(test_fold)
    if return_array:
        x = np.concatenate([x_train, x_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)
        return ps, [x, y]
    return ps

