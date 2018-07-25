from . import dataset
from . import network
from . import trainer
from . import logger
from . import visualizer
from . import postprocess

import glob
import os
import os.path
import warnings

import chainer
import numpy

def parse_index_file(filename, ratio = None):
    """Parse index file
    
    Args:
        filename (str, None): A filename will be loaded.
        ratio (float, optional): A separating ratio (Default:None)
    
    Returns:
        str, float: Loaded indices
    """


    indices = []
    if filename is None:
        return ratio

    with open(filename, 'r') as fp:
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

def deprecated():
    def _deprecated(func):
        warnings.warn(
            'Invoked function {} is deprecated.'.format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2
        )
        return func
    return _deprecated

def get_field(model, names):
    if len(names) == 0:
        return model
    return get_field(getattr(model, names[0]), names[1:])