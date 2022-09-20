import os
import logging
import numpy as np
import torch
import random 
import torch.backends.cudnn as cudnn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def print_log(message):
    print(message)
    logging.info(message)

def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_dataset(config):
    from API import load_data
    return load_data(**config)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
