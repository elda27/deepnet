from deepnet.core.registration import register_process
import chainer
import chainer.functions as F

@register_process('loss.total_softmax_cross_entropy')
def total_softmax_cross_entropy(x, t, normalize=True):
    """Softmax cross entropy between single channel and multi-channel label.
    
    Args:
        x (chainer.Variable): Single channel label
        t (chainer.Variable): Multi-channel label
        normalize (bool, optional): Defaults to True. If true, the entropy will be normalization.
    
    Returns:
        chainer.Variable: Calculation result
    """


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
