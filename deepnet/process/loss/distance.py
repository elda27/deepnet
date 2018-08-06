from deepnet.core.registration import register_process
import chainer.functions as F
import functools

@register_process('loss.euclidean_distance')
def euclidean_distance(x, t):
    linear_shape = (x.shape[0], functools.reduce(lambda x,y: x * y, x.shape[1:]))
    return F.mean(F.sqrt(F.batch_l2_norm_squared(F.reshape(x - t, linear_shape))))
