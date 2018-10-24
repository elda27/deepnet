from deepnet.core.registration import register_process
import chainer
import chainer.functions as F
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

    d = F.minimum(d, 1e-8)

    return 1.0 / d
