import chainer
import chainer.functions as F
import numpy as np
from deepnet import utils
try:
    import cupy as cp
except ImportError:
    pass
from functools import reduce

def constrain_kernel(network):
    n_kernels = 0
    norm = None
    for node in network.updatable_node:
        for lname, layer in node.model.layers:
            if isinstance(layer, (chainer.links.ConvolutionND, chainer.links.DeconvolutionND)):
                n_kernels += 1
                if norm is None:
                    norm = F.batch_l2_norm_squared(layer.W)
                else:
                    norm += F.batch_l2_norm_squared(layer.W)
    return norm / n_kernels

def euclidean_distance(x, t):
    linear_shape = (x.shape[0], reduce(lambda x,y: x * y, x.shape[1:]))
    return F.mean(F.batch_l2_norm_squared(F.reshape(x - t, linear_shape)))

def total_softmax_cross_entropy(x, t, normalize=False):
    assert len(x) == len(t)
    num_channels = len(t)
    xs = F.expand_dims(x, axis=1)
    ts = F.expand_dims(_make_overlap(t), axis=1)

    def make_filled(img, fill_value):
        xp = cp if isinstance(img.array, cp.ndarray) else np
        return xp.ones((img.shape[0], 1) + img.shape[2:], xp.float32) * 1e-3

    bg_xs = make_filled(xs[0], 1e-3)
    
    loss = [ F.softmax_cross_entropy(F.concat((bg_xs, xs[i]), axis=1), ts[i], normalize=normalize) for i in range(xs.shape[0]) ]
    return sum(loss) / xs.shape[0]

def _expand_background(labels):
    xp = cp if isinstance(labels.array, cp.ndarray) else np
    empty_label = xp.ones((labels.shape[0], 1) + labels.shape[2:], xp.float32) * 1e-3
    empty_label = chainer.Variable(empty_label)
    labels = F.concat((empty_label, chainer.Variable(labels.data.astype(xp.float32))), axis=1)
    return labels

def _make_overlap(labels):
    labels = _expand_background(labels)
    return F.argmax(labels, axis=1)
