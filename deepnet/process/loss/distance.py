from deepnet.core.registration import register_process
import chainer.functions as F
import functools


@register_process('loss.euclidean_distance')
def euclidean_distance(x, t, normalize=False):
    l2_norm = F.sqrt(F.batch_l2_norm_squared(x-t))
    if normalize:
        linear_shape = functools.reduce(lambda x, y: x * y, x.shape[1:])
        l2_norm = l2_norm / linear_shape
    return F.mean(l2_norm)
