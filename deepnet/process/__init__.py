import chainer
import chainer.functions as F
import math
import numpy as np
NO_CUPY = False
try:
    import cupy as cp
except ImportError:
    NO_CUPY = True
    pass
import functools
from itertools import cycle

from deepnet import utils

_registed_process = {
    'chainer.sigmoid': F.sigmoid,
    'chainer.softmax': F.softmax,
    'chainer.transpose': F.transpose,
    'chainer.expand_dims': F.expand_dims,
    'chainer.sigmoid_cross_entropy': F.sigmoid_cross_entropy,
    'chainer.softmax_cross_entropy': F.softmax_cross_entropy,
    'chainer.batch_l2_norm_squared': F.batch_l2_norm_squared,
}

def register_process(name = None):
    def _register_process(func):
        if name is None:
            assert func.__name__ not in _registed_process
            _registed_process[func.__name__] = func
        else:
            assert name not in _registed_process
            _registed_process[name] = func
        return func
    return _register_process

Attributes = {}


def set_gpu_id(gpu_id):
    global Attributes
    Attributes['gpu_id'] = gpu_id

@register_process()
def binary(*images, threshold=None):
    assert threshold is not None, '"threshold" is requirement argument'

    bin_images = []
    for image in images:
        assert not isinstance(image, chainer.Variable), 'Unsupported chainer.Variable.'
        if isinstance(image, list):
            bin_images.append([(img > threshold).astype(np.int32) for img in image ])
        else:
            bin_images.append(image > threshold)
    return bin_images

@register_process()
def normalize(*images):
    results = []
    for img in images:
        img = utils.unwrapped(img)
        amax = np.amax(img)
        amin = np.amin(img)
        results.append((img - amax) / (amax - amin + 1e-8))
    return results[0] if len(results) == 1 else results

@register_process()
def blend_image(*image_pairs):
    result_image = []
    for fore, back in zip(image_pairs[::2], image_pairs[1::2]):
        fore = utils.unwrapped(fore)
        back = utils.unwrapped(back)
        if back.shape[1] > fore.shape[1]:
            fore = F.repeat(fore, back.shape[1] // fore.shape[1], axis=1)
        elif back.shape[1] < fore.shape[1]:
            back = F.repeat(back, fore.shape[1] // back.shape[1], axis=1)
        result_image.append(normalize(fore) * 0.5 + normalize(back) * 0.5)
    return result_image

@register_process()
def make_overlap_label(*images):
    result_image = []
    for image in images:
        img = utils.unwrapped(image)
        batch_image = []
        
        index_img = np.argmax(
            np.concatenate((np.ones( (img.shape[0], 1) + img.shape[2:], dtype=img.dtype) * 1e-1, img), axis=1), 
            axis=1
            )
        result_image.append(map_index_label(index_img))
    return result_image

colors = [
    (233, 66, 59),   # Red
    (222, 29, 98),   # Pink
    (150, 41, 72),   # Purple
    (101, 59, 79),   # Deep purple
    (68, 82, 177),   # Indigo
    (65, 151, 239),  # Blue
    (62, 170, 241),  # Light blue
    (66, 188, 211),  # Cyan
    (50, 150, 136),  # Teal
    (91, 175, 87),   # Green
    (145, 195, 85),  # Light green
    (208, 220, 78),  # Lime
    (252, 235, 83),  # Yellow
    (249, 193, 51),  # Amber
    (248, 151, 40),  # Orange
    (244, 86, 45),   # Deep Orange
    (117, 85, 73),   # Brown
    (158, 158, 158), # Gray
    (99, 125, 138),  # Blue Gray
]

@register_process()
def map_index_label(img, colors=colors):
    """Mapping color to index image
    
    Args:
        img (images): Input images aligned by (N, 1, S, ...) or (N, S, ...) (N means N sample so it is same as batch size, 
                      S mean shape of image).
        colors (list, optional): color of input images, Defaults to colors.
    
    Returns:
        numpy.ndarray: color images (N, 3, S), S is same as input image.
    """

    if img.ndim == 2:
        return map_index_label_2d(img)
    
    img = np.squeeze(utils.unwrapped(img))

    uniques = list(np.unique(img))
    if 0 in uniques:
        uniques.remove(0)

    result = np.zeros((3, ) + img.shape, dtype=np.uint8)
    for uid, color in zip(uniques, cycle(colors)):
        mask = (img == uid)
        result[0, mask] = color[0]
        result[1, mask] = color[1]
        result[2, mask] = color[2]
    
    return np.transpose(result, (1, 0, 2, 3))

@register_process
def map_index_label_2d(img, colors=colors):
    assert img.ndim == 2, 'Actual: {} (Shape: {})'.format(img.ndim, img.shape)

    uniques = list(np.unique(img))
    if 0 in uniques:
        uniques.remove(0)
    
    result = np.zeros((3, img.shape[0], img.shape[1]), dtype=np.uint8)
    for uid, color in zip(uniques, cycle(colors)):
        mask = (img == uid)
        result[0, mask] = color[0]
        result[1, mask] = color[1]
        result[2, mask] = color[2]
    return result

@register_process()
def cast_type(x, dtype):
    return chainer.Variable(x.data.astype(dtype))

@register_process()
def bias(x, multiply=1.0, bias_=1.0):
    return x * multiply + bias_

@register_process()
def apply_gaussian_noise(x, sigma=1.0, clip=None, device=-1):
    ones = None
    if device >= 0:
        ones = cp.ones_like(x.data)
        zeros = cp.ones_like(x.data)
    else:
        ones = np.ones_like(x.data)
    ones = chainer.Variable(ones)

    if clip is None:
        result = F.gaussian(x, math.log(sigma) * ones)
    else: #clip is not None:
        min_value, max_value = clip
        result = x + F.gaussian(0 * ones, math.log(sigma) * ones)
        result = F.clip(result, min_value, max_value)

    return result

@register_process()
def to_cpu(*input_list):
    output_list = []
    for input_ in input_list:
        if isinstance(input_, list):
            input_ = [ F.expand_dims(chainer.Variable(i), axis=0) for i in input_ ]
            input_ = F.concat(input_, axis=0)
        else:
            input_ = chainer.Variable(input_)
        output_list.append(input_)
    return output_list

@register_process()
def to_gpu(*input_list):
    if NO_CUPY:
        return to_cpu(*input_list)
        
    output_list = []
    for input_ in input_list:
        if isinstance(input_, list):
            #input_ = [ F.expand_dims(chainer.Variable(i), axis=0) for i in input_ ]
            #input_ = F.concat(input_, axis=0)
            input_ = chainer.Variable(np.concatenate([ np.expand_dims(i, axis=0) for i in input_ ], axis=0))
        elif isinstance(input_, chainer.Variable):
            input_ = F.copy(input_, -1)
        else:
            input_ = chainer.Variable(input_)
        input_.to_gpu()
        output_list.append(input_)
    return output_list

__operation_list = {
    '+': lambda x, y: x + y,
    '-': lambda x, y: x - y,
    '*': lambda x, y: x * y,
    '/': lambda x, y: x / y,
}

@register_process()
def reduce(*input, operation='+', weights=None):
    if weights is None:
        weights = [ 1.0 for _ in range(len(input)) ]

    operation = __operation_list[operation]
    input_iter = iter(zip(input, weights))

    x0, w0 = next(input_iter)
    y = x0 * w0
    for x, w in input_iter:
        y = operation(x * w, y)

    return y

@register_process('loss.constrain_kernel')
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

@register_process('loss.euclidean_distance')
def euclidean_distance(x, t):
    linear_shape = (x.shape[0], functools.reduce(lambda x,y: x * y, x.shape[1:]))
    return F.mean(F.batch_l2_norm_squared(F.reshape(x - t, linear_shape)))

@register_process('loss.total_softmax_cross_entropy')
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
