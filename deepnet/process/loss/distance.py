from deepnet.core.registration import register_process
import chainer.functions as F
import functools


@register_process('loss.euclidean_distance')
def euclidean_distance(x, t, normalize=False, reduce='mean', sqrt=F.sqrt):
    l2_norm = sqrt(F.batch_l2_norm_squared(x - t))
    if normalize:
        linear_shape = functools.reduce(lambda x, y: x * y, x.shape[1:])
        l2_norm = l2_norm / linear_shape

    if reduce:
        if reduce == 'mean':
            return F.mean(l2_norm)
        elif reduce == 'sum':
            return F.sum(l2_norm)
        else:
            raise AttributeError('Unknown reduction type: ' + reduce)
    else:
        return l2_norm


@register_process('loss.l2_norm_distance')
def l2_norm_distance(x, t, reduce='sum'):
    l2_norm = (x - t) ** 2
    if reduce == 'mean':
        return F.mean(l2_norm)
    elif reduce == 'sum':
        return F.sum(l2_norm)
    elif reduce == 'no':
        return l2_norm
    else:
        raise AttributeError('Unknown reduction type: ' + reduce)
