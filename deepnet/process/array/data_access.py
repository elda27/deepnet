from corenet import declare_node_type
from deepnet.core.registration import register_process
from deepnet.core import config
import chainer
import chainer.functions as F
import numpy as np
import random


@register_process()
def get_element(*input, index=0, axis=1):
    return [F.rollaxis(i, axis)[index] for i in input]


@register_process()
def get_random_element(*input, axis=1):
    n = max([i.shape[axis] for i in input])
    return get_element(*input, index=random.randint(0, n), axis=axis)


@declare_node_type('iterable')
@register_process()
def volume_to_slice(*input):
    for i in zip(*input):
        yield i


@register_process()
def merge_slice_to_volume(*input):
    output = []
    for i in input:
        output.append(np.array(i))
    return output
