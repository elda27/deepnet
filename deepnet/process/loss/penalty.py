from deepnet.core.registration import register_process
import chainer
import chainer.functions as F
from chainer.backends import cuda
import numpy as np
import functools

from .distance import euclidean_distance


@register_process('loss.constrain_skip_connection')
def constrain_skip_connection(x, t, normalize=False):
    if x.shape == t.shape:
        d = euclidean_distance(x, t, normalize=normalize)
    else:
        ds = []
        for i in range(t.shape[1]):
            ds.append(euclidean_distance(
                x,
                F.expand_dims(t[:, i, ...], axis=1),
                normalize=normalize))
        d = sum(ds)

    xp = cuda.get_array_module(d)
    d = F.minimum(d, xp.array([1e-8], dtype=xp.float32))

    return 1.0 / d
