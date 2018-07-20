import chainer
import chainer.cuda
from chainer import functions as F
from chainer import links as L
from chainer.utils import argument
import numpy as np

def bayesian_dropout(x, ratio=.5, **kwargs):
    """bayesian_dropout(x, ratio=.5)
    Drops elements of input variable randomly.
    This function drops input elements randomly with probability ``ratio`` and
    scales the remaining elements by factor ``1 / (1 - ratio)``. 
    .. warning::
       ``train`` argument is not supported anymore since v2.
       Instead, use ``chainer.using_config('train', boolean)``.
       See :func:`chainer.using_config`.
    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)` -shaped float array.
        ratio (float):
            Dropout ratio. The ``ratio`` must be ``0.0 <= ratio < 1.0``.
    Returns:
        ~chainer.Variable: Output variable.
    See the paper by A. Kendall: `Bayesian SegNet: Model Uncertainty \
    in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding \
    <https://arxiv.org/abs/1511.02680>`_.
    """

    argument.check_unexpected_kwargs(
        kwargs, train='train argument is not supported anymore. '
        'Use chainer.using_config')
    argument.assert_kwargs_empty(kwargs)

    if chainer.config.train:
        return F.Dropout(ratio).apply((x,))[0]
    else:
        return F.Dropout(ratio).apply((x,))[0]

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_filter_2d(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return filter

    
def get_upsampling_filter_3d(size):
    """Make a 3D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor) * \
             (1 - abs(og[2] - center) / factor)
    return filter
    
def crop(t, shape, n_dim = 2):
    '''
      Cropping t by x.shape
    '''
    if n_dim == 2:
        left   = (t.shape[2] - shape[2]) // 2
        top    = (t.shape[3] - shape[3]) // 2
        right  = left + shape[2]
        bottom = top  + shape[3]
        assert left >= 0 and top >= 0 and right < t.shape[2] and bottom < t.shape[3], \
            'Cropping image is less shape than input shape.\nInput shape:{}, Cropping shape:{}, (L,R,T,B):({},{},{},{})'.format(t.shape, shape, left, right, top, bottom)
        return t[:, :, left:right, top:bottom]
    raise NotImplementedError('3D or Nd cropping is not inmplemented.')

class CBR(chainer.Chain):
    dropout = dict(
        bayesian=bayesian_dropout,
        dropout=F.dropout,
        none=None,
    )

    def __init__(self, n_dims, in_ch, out_ch, ksize=4, stride=2, bn=True, sample='down', activation=F.relu, dropout='dropout'):
        self.use_bn = bn
        self.activation = activation
        self.dropout = None if dropout in CBR.dropout else CBR.dropout[dropout]

        w = chainer.initializers.Normal(0.02)

        super().__init__()
        with self.init_scope():
            if sample == 'down':
                self.c = L.ConvolutionND(n_dims, in_ch, out_ch, ksize=ksize, stride=stride, pad=1, initialW=w)
            else:
                self.c = L.DeconvolutionND(n_dims, in_ch, out_ch, ksize=ksize, stride=stride, pad=1, initialW=w)
            if bn:
                self.bn = L.BatchNormalization(out_ch)

        
    def __call__(self, x):
        h = self.c(x)
        if self.use_bn:
            h = self.bn(h)
        if self.dropout:
            h = F.dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h

        
from deepnet.network.init import register_network
import deepnet.utils.network 

@register_network('patch_inference')
class PatchInferencer(deepnet.utils.network.IterableProcessor):
    def __init__(self, shape):
        self.patch_shape = shape
        self.index = [0] * len(shape)
        self.output = None

    def __call__(self, array):
        if self.output is None:
            xp = chainer.cuda.get_array_module(array)
            self.output = xp.zeros_like(array)
            if isinstance(array, chainer.Variable):
                self.output = chainer.Variable(self.output)

        start = [ self.index[i] * self.patch_shape for i in range(len(self.patch_shape))]
        end = [ max(self.index[i] * self.patch_shape, array.shape[i]) for i in range(len(self.patch_shape))]
        slices = tuple(map(slice, start, end))
        self.current_slice = slices
        return array[slices]

    def insert(self, array):
        self.output[self.current_slice] = array

    def get_output(self):
        output = self.output
        self.output = None
        return output

