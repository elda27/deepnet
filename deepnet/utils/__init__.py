from . import dataset
from . import network
from . import trainer
from . import logger
from . import visualizer

import glob
import os
import os.path

import chainer
import numpy

def parse_index_file(filename):
    indices = []
    with open(filename) as fp:
        for line in fp.readlines():
            indices.append(line.strip())
    return indices

def unwrapped(var):
    if isinstance(var, chainer.Variable):
        var = chainer.functions.copy(var, -1).data
    if isinstance(var, numpy.ndarray) and var.ndim == 0:
        var = float(var)
    return var


def batch_to_vars(batch):
    # batch to vars
    input_vars = [ dict() for elem in batch[0] ]
    for elem in batch:              # loop about batch
        for i, stage_input in enumerate(elem):    # loop about stage input
            for name, input_ in stage_input.items():
                input_vars[i].setdefault(name, []).append(input_)
    return input_vars

def get_log_dir(log_root_dir, log_index):
    if log_index is None:
        return log_root_dir
    log_glob = list(glob.glob(os.path.join(log_root_dir, str(log_index) + '-*')))
    assert len(log_glob) == 1
    return log_glob[0]
