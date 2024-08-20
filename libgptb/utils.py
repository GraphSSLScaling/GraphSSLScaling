from typing import *
import os
import torch
import random
import numpy as np

import importlib
import logging
import datetime
import sys


def get_executor(config, model, data_feature):
    """
    according the config['executor'] to create the executor

    Args:
        config(ConfigParser): config
        model(AbstractModel): model

    Returns:
        AbstractExecutor: the loaded executor
    """
    # getattr(importlib.import_module('libgptb.executors'),
    #                     config['executor'])(config, model, data_feature)
    try:
        return getattr(importlib.import_module('libgptb.executors'),
                       config['executor'])(config, model, data_feature)
    except AttributeError:
        raise AttributeError('executor is not found')


def get_model(config, data_feature):
    """
    according the config['model'] to create the model

    Args:
        config(ConfigParser): config
        data_feature(dict): feature of the data

    Returns:
        AbstractModel: the loaded model
    """
    if config['task'] == 'GCL' or config['task'] == 'SSGCL' or config['task'] == 'SGC':
        try:
            return getattr(importlib.import_module('libgptb.model'),
                           config['model'])(config, data_feature)
        except AttributeError:
            raise AttributeError('model is not found')
    else:
        raise AttributeError('task is not found')


def get_evaluator(config):
    """
    according the config['evaluator'] to create the evaluator

    Args:
        config(ConfigParser): config

    Returns:
        AbstractEvaluator: the loaded evaluator
    """
    try:
        return getattr(importlib.import_module('libgptb.evaluator'),
                       config['evaluator'])(config)
    except AttributeError:
        raise AttributeError('evaluator is not found')


def get_logger(config, name=None):
    """
    获取Logger对象

    Args:
        config(ConfigParser): config
        name: specified name

    Returns:
        Logger: logger
    """
    log_dir = './libgptb/log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(config.config)
    log_filename = '{}-{}-{}-{}-{}-{}.log'.format(config['model'],config['dataset'],config.get("ratio",1),
                                            config['config_file'], config['exp_id'], get_local_time())

    logfilepath = os.path.join(log_dir, log_filename)


    logger = logging.getLogger(name)

    log_level = config.get('log_level', 'INFO')

    if log_level.lower() == 'info':
        level = logging.INFO
    elif log_level.lower() == 'debug':
        level = logging.DEBUG
    elif log_level.lower() == 'error':
        level = logging.ERROR
    elif log_level.lower() == 'warning':
        level = logging.WARNING
    elif log_level.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logfilepath)
    file_handler.setFormatter(formatter)

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('Log directory: %s', log_dir)
    return logger


def get_local_time():
    """
    获取时间

    Return:
        datetime: 时间
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur


def ensure_dir(dir_path):
    """Make sure the directory exists, if it does not exist, create it.

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)




def set_random_seed(seed):
    """
    重置随机数种子

    Args:
        seed(int): 种子数
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



def normalize(s):
    return (s.max() - s) / (s.max() - s.mean())




"""
store the arguments can be modified by the user
"""
import argparse

general_arguments = {
    "gpu": {
        "type": "bool",
        "default": None,
        "help": "whether use gpu"
    },
    "gpu_id": {
        "type": "int",
        "default": None,
        "help": "the gpu id to use"
    },
    "train_rate": {
        "type": "float",
        "default": None,
        "help": "the train set rate"
    },
    "eval_rate": {
        "type": "float",
        "default": None,
        "help": "the validation set rate"
    },
    "batch_size": {
        "type": "int",
        "default": None,
        "help": "the batch size"
    },
    "learning_rate": {
        "type": "float",
        "default": None,
        "help": "learning rate"
    },
    "max_epoch": {
        "type": "int",
        "default": None,
        "help": "the maximum epoch"
    },
    "dataset_class": {
        "type": "str",
        "default": None,
        "help": "the dataset class name"
    },
    "executor": {
        "type": "str",
        "default": None,
        "help": "the executor class name"
    },
    "evaluator": {
        "type": "str",
        "default": None,
        "help": "the evaluator class name"
    },
}

hyper_arguments = {
    "gpu": {
        "type": "bool",
        "default": None,
        "help": "whether use gpu"
    },
    "gpu_id": {
        "type": "int",
        "default": None,
        "help": "the gpu id to use"
    },
    "train_rate": {
        "type": "float",
        "default": None,
        "help": "the train set rate"
    },
    "eval_rate": {
        "type": "float",
        "default": None,
        "help": "the validation set rate"
    },
    "batch_size": {
        "type": "int",
        "default": None,
        "help": "the batch size"
    }
}


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('bool value expected.')


def str2float(s):
    if isinstance(s, float):
        return s
    try:
        x = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError('float value expected.')
    return x


def add_general_args(parser):
    for arg in general_arguments:
        if general_arguments[arg]['type'] == 'int':
            parser.add_argument('--{}'.format(arg), type=int,
                                default=general_arguments[arg]['default'], help=general_arguments[arg]['help'])
        elif general_arguments[arg]['type'] == 'bool':
            parser.add_argument('--{}'.format(arg), type=str2bool,
                                default=general_arguments[arg]['default'], help=general_arguments[arg]['help'])
        elif general_arguments[arg]['type'] == 'float':
            parser.add_argument('--{}'.format(arg), type=str2float,
                                default=general_arguments[arg]['default'], help=general_arguments[arg]['help'])
        elif general_arguments[arg]['type'] == 'str':
            parser.add_argument('--{}'.format(arg), type=str,
                                default=general_arguments[arg]['default'], help=general_arguments[arg]['help'])
        elif general_arguments[arg]['type'] == 'list of int':
            parser.add_argument('--{}'.format(arg), nargs='+', type=int,
                                default=general_arguments[arg]['default'], help=general_arguments[arg]['help'])