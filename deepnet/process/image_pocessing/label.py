from deepnet.core.registration import register_process
from matplotlib.pyplot import get_cmap

import chainer
from chainer.backends import cuda

import numpy as np
import cupy as cp


@register_process()
def map_label(*input, index_map={}):
    assert len(index_map) != 0
    outputs = []
    for image in input:
        xp = cuda.get_array_module(image)
        output = xp.copy(image.data)
        for pairs in index_map.items():
            source, dest = map(int, pairs)
            mask = image.data == source
            output[mask] = dest
        outputs.append(chainer.Variable(output))
    return outputs


@register_process()
def make_overlap_label(*images, color='tab10'):
    cmap = get_cmap(color)
    result_image = []
    for image in images:
        img = utils.unwrapped(image)

        index_img = np.argmax(
            np.concatenate(
                (np.ones((img.shape[0], 1) + img.shape[2:], dtype=img.dtype) * 1e-1, img), axis=1),
            axis=1
        )
        result_image.append(map_index_label(index_img))
    return result_image


@register_process()
def map_index_label(label):
    """Mapping color to index image

    Args:
        label (images): Input images aligned by (N, 1, S, ...) or 
                        (N, S, ...) (N means N sample so it is same as batch size, 
                        S mean shape of image).
        color (list, optional): color of input images, Defaults to colors.

    Returns:
        numpy.ndarray: color images (N, 3, S, ...), S is same as input image.
    """

    color = 'tab10'
    fold = 10

    indices = list(np.unique(label))
    indices.remove(0)
    color_image = np.tile(
        np.expand_dims(
            np.zeros_like(label),
            axis=label.ndim
        ),
        (1,) * label.ndim + (3,)
    )

    for i in reversed(indices):
        color = cmap(((i - 1) % fold) / fold)
        mask = label == i
        r = np.expand_dims(mask * color[0] * 255, axis=mask.ndim)
        g = np.expand_dims(mask * color[1] * 255, axis=mask.ndim)
        b = np.expand_dims(mask * color[2] * 255, axis=mask.ndim)
        color_mask = np.tile(
            np.expand_dims(
                np.logical_not(mask),
                axis=mask.ndim
            ),
            (1,) * mask.ndim + (3,)
        )
        color_image = color_image * color_mask * \
            np.concatenate((r, g, b), axis=mask.ndim)

    return np.transpose(color_image, (1, 0, 2, 3))


@register_process()
def label_to_probability(label, n_channel=None):
    xp = cuda.get_array_module(label)
    if isinstance(label, chainer.Variable):
        label = label.data

    if label.dtype.kind != 'i':
        label = label.astype(xp.int32)

    if xp == np:
        unique_indexes = np.unique(label)
    elif xp == cp:
        unique_indexes = unique(label)

    if n_channel is None:
        n_channel = len(unique_indexes)

    probs = xp.zeros(
        (n_channel, ) + label.shape,
        dtype=xp.float32
    )
    for i, index in enumerate(unique_indexes):
        probs[i] = (label == index).astype(xp.float32)
    probs = xp.rollaxis(probs, 1, 1)

    return chainer.Variable(probs)


def unique(ar, return_index=False, return_inverse=False,
           return_counts=False, axis=None):
    """ This implementation is copied from cupy 6.0.0.a
    Find the unique elements of an array.
    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements:
    * the indices of the input array that give the unique values
    * the indices of the unique array that reconstruct the input array
    * the number of times each unique value comes up in the input array
    Args:
        ar(array_like): Input array. This will be flattened if it is not
            already 1-D.
        return_index(bool, optional): If True, also return the indices of `ar`
            (along the specified axis, if provided, or in the flattened array)
            that result in the unique array.
        return_inverse(bool, optional): If True, also return the indices of the
            unique array (for the specified axis, if provided) that can be used
            to reconstruct `ar`.
        return_counts(bool, optional): If True, also return the number of times
            each unique item appears in `ar`.
        axis(int or None, optional): Not supported yet.
    Returns:
        cupy.ndarray or tuple:
            If there are no optional outputs, it returns the
            :class:`cupy.ndarray` of the sorted unique values. Otherwise, it
            returns the tuple which contains the sorted unique values and
            followings.
            * The indices of the first occurrences of the unique values in the
              original array. Only provided if `return_index` is True.
            * The indices to reconstruct the original array from the
              unique array. Only provided if `return_inverse` is True.
            * The number of times each of the unique values comes up in the
              original array. Only provided if `return_counts` is True.
    .. seealso:: :func:`numpy.unique`
    """
    if axis is not None:
        raise NotImplementedError('axis option is not supported yet.')

    ar = cp.asarray(ar).flatten()

    if return_index or return_inverse:
        perm = ar.argsort()
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    mask = cp.empty(aux.shape, dtype=cp.bool_)
    mask[0] = True
    mask[1:] = aux[1:] != aux[:-1]

    ret = aux[mask]
    if not return_index and not return_inverse and not return_counts:
        return ret

    ret = ret,
    if return_index:
        ret += perm[mask],
    if return_inverse:
        imask = cp.cumsum(mask) - 1
        inv_idx = cp.empty(mask.shape, dtype=cp.intp)
        inv_idx[perm] = imask
        ret += inv_idx,
    if return_counts:
        nonzero = cp.nonzero(mask)[0]
        idx = cp.empty((nonzero.size + 1,), nonzero.dtype)
        idx[:-1] = nonzero
        idx[-1] = mask.size
        ret += idx[1:] - idx[:-1],
    return ret
