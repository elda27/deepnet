from deepnet.core.registration import register_process
from deepnet.core import config
import chainer
from chainer import functions as F
import numpy as np
import cupy as cp


@register_process()
def expand_dims(*input, axis=1):
    output = []
    for i in input:
        xp = chainer.cuda.get_array_module(i)
        output.append(xp.expand_dims(i, axis=axis))
    return output


@register_process()
def batch_reshape(x, shape):
    shape = tuple(map(int, shape))
    return F.reshape(x, (x.shape[0],) + shape)


@register_process()
def cast_type(x, dtype):
    xp = chainer.cuda.get_array_module(x)
    if isinstance(x, xp.ndarray):
        return x.astype(dtype)
    else:
        return chainer.Variable(x.data.astype(dtype))


@register_process()
def bias(x, multiply=1.0, bias_=1.0):
    return x * multiply + bias_


@register_process()
def get_latent_representation(*_, source):
    from deepnet.core.network.build import get_process

    def get_field(var, fields):
        if len(fields) == 0:
            return var
        return get_field(getattr(var, fields[0]), fields[1:])

    accessors = source.split('.')
    proc = get_field(get_process(accessors[0]), accessors[1:-1])

    return proc.stores[accessors[-1]]


@register_process()
def concat(*x, axis=1):
    return F.concat(tuple(x), axis=axis)


@register_process()
def split(x, pos, axis=1):
    first = tuple(slice(pos) if i == axis else slice(x.shape[i])
                  for i in range(x.ndim))
    last = tuple(slice(pos, x.shape[i]) if i == axis else slice(x.shape[i])
                 for i in range(x.ndim))
    return x[first], x[last]
