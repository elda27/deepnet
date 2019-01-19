from deepnet.core.registration import register_process
import math
import chainer
from chainer import cuda, functions as F

@register_process()
def apply_gaussian_noise(x, sigma=1.0, clip=None):
    """Apply gaussian noise to n-dimensional image.
    
    Args:
        x (chainer.Variable): Input n-dimensional image.
        sigma (float, optional): Defaults to 1.0. The variation of gaussian distribution.
        clip (list[float], optional): Defaults to None. A intensity will be clipping this tuple.
    
    Returns:
        chainer.Variable: Images applied gaussin noise
    """

    xp = cuda.get_array_module(x)
    ones = chainer.Variable(xp.ones_like(x.data))

    if clip is None:
        result = F.gaussian(x, math.log(sigma) * ones)
    else: #clip is not None:
        min_value, max_value = clip
        result = x + F.gaussian(0 * ones, math.log(sigma) * ones)
        result = F.clip(result, min_value, max_value)

    return result


@register_process()
def apply_uniform_noise(x, scale=1.0, bias=0.0, clip=None):
    """Apply uniform noise to n-dimensional image.

    Args:
        x (chainer.Variable): Input n-dimensional image.
        scale (float, optional): Defaults to 1.0. A scale of noise.
        bias (list[float], optional): Defaults to 0.0. A bias of noise.

    Returns:
        chainer.Variable: Images applied uniform noise
    """

    xp = cuda.get_array_module(x)
    noise = chainer.Variable(xp.random.rand(x.data))

    result = x + noise
    if clip is not None:
        min_value, max_value = clip
        result = F.clip(result, min_value, max_value)

    return result