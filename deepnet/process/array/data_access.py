from deepnet.core.registration import register_process
from deepnet.core import config
import chainer
import chainer.functions as F
import random


@register_process()
def get_element(*input, index=0, axis=1):
    return [F.rollaxis(i, axis)[index] for i in input]


@register_process()
def get_random_element(*input, axis=1):
    n = max([i.shape[axis] for i in input])
    return get_element(*input, index=random.randint(0, n), axis=axis)
