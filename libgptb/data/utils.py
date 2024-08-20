import importlib
import numpy as np
from torch.utils.data import DataLoader
import copy

from libgptb.data.batch import Batch, BatchPAD


def get_dataset(config):
    """
    according the config['dataset_class'] to create the dataset

    Args:
        config(ConfigParser): config

    Returns:
        AbstractDataset: the loaded dataset
    """

    try:
        return getattr(importlib.import_module('libgptb.data.dataset'),
                       config['dataset_class'])(config)
    except AttributeError:
        print("Error Attribute")
        try:
            return getattr(importlib.import_module('libgptb.data.dataset.dataset_subclass'),
                           config['dataset_class'])(config)
        except AttributeError:
            raise AttributeError('dataset_class is not found')

