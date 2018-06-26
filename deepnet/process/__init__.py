import chainer
import chainer.functions as F
import math
import numpy as np
try:
    import cupy as cp
except ImportError:
    pass
from functools import reduce
from itertools import cycle

from deepnet import utils
from . import loss

Attributes = {}

def set_gpu_id(gpu_id):
    global Attributes
    Attributes['gpu_id'] = gpu_id

def normalize(*images):
    results = []
    for img in images:
        img = utils.unwrapped(img)
        amax = np.amax(img)
        amin = np.amin(img)
        results.append((img - amax) / (amax - amin + 1e-8))
    return results[0] if len(results) == 1 else results

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

def make_overlap_label(*images):
    result_image = []
    for image in images:
        img = utils.unwrapped(image)
        batch_image = []
        for i in range(img.shape[0]):
            index_img = np.argmax(np.concatenate((np.ones((1,) + img[i].shape[1:], dtype=img.dtype) * 1e-1, img[i]), axis=0), axis=0)
            batch_image.append(np.expand_dims(map_index_label(index_img.T), axis=0))
        result_image.append(np.concatenate(batch_image, axis=0))
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

def map_index_label(img, colors=colors):
    assert img.ndim == 2, 'Actual: {}'.format(img.ndim)
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

def cast_type(x, dtype):
    return chainer.Variable(x.data.astype(dtype))

def bias(x, multiply=1.0, bias_=1.0):
    return x * multiply + bias_

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

def to_gpu(*input_list):
    output_list = []
    for input_ in input_list:
        if isinstance(input_, list):
            input_ = [ F.expand_dims(chainer.Variable(i), axis=0) for i in input_ ]
            input_ = F.concat(input_, axis=0)
        elif isinstance(input_, chainer.Variable):
            input_ = F.copy(input_, -1)
        else:
            input_ = chainer.Variable(input_)
        input_.to_gpu()
        output_list.append(input_)
    return output_list
