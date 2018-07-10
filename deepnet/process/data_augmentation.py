import deepnet.auto_path
import chainer.cuda
import numpy as np
import scipy.ndimage as ndimage

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cu_ndimage
except:
    pass


def rotate(input, angle, axes=(1, 0), reshape=True, output=None, order=None, mode='constant', cval=0.0, prefilter=True):
    xp = chainer.cuda.get_array_module(input)
    if xp == np:
        return ndimage.rotate(input, angle, axes, reshape, output, order, mode, cval, prefilter)
    else:
        return cu_ndimage.rotate(input, angle, axes, reshape, output, order, mode, cval, prefilter)

def shift(input, shift, output=None, order=None, mode='constant', cval=0.0, prefilter=True):
    xp = chainer.cuda.get_array_module(input)
    if xp == np:
        return ndimage.shift(input, shift, output, order, mode, cval, prefilter)
    else:
        return cu_ndimage.shift(input, shift, output, order, mode, cval, prefilter)

def zoom(input, zoom, output=None, order=None, mode='constant', cval=0.0, prefilter=True):
    xp = chainer.cuda.get_array_module(input)
    if xp == np:
        return ndimage.zoom(input, zoom, output, order, mode, cval, prefilter)
    else:
        return cu_ndimage.zoom(input, zoom, output, order, mode, cval, prefilter)

def affine_transform(input, matrix, offset=0.0, output_shape=None, output=None, order=None, mode='constant', cval=0.0, prefilter=True):
    xp = chainer.cuda.get_array_module(input)
    if xp == np:
        return ndimage.affine_transform(input, matrix, offset, output_shape, output, order, mode, cval, prefilter)
    else:
        return cu_ndimage.affine_transform(input, matrix, offset, output_shape, output, order, mode, cval, prefilter)


