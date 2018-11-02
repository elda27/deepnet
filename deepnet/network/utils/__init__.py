import chainer
import chainer.cuda
from chainer import functions as F
from chainer import links as L
from chainer.functions.noise.dropout import Dropout
from chainer.utils import argument
import numpy as np
import functools
import random


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
        return Dropout(ratio).apply((x,))[0]
    else:
        return Dropout(ratio).apply((x,))[0]

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


def crop(t, shape, n_dim=2):
    '''
      Cropping t by x.shape
    '''
    if t.shape == shape:
        return t

    if n_dim == 2:
        left = (t.shape[2] - shape[2]) // 2
        top = (t.shape[3] - shape[3]) // 2
        right = left + shape[2]
        bottom = top + shape[3]
        assert left >= 0 and top >= 0 and right < t.shape[2] and bottom < t.shape[3], \
            'Cropping image is less shape than input shape.\nInput shape:{}, Cropping shape:{}, (L,R,T,B):({},{},{},{})'.format(
                t.shape, shape, left, right, top, bottom)
        return t[:, :, left:right, top:bottom]
    if n_dim == 3:
        left = (t.shape[2] - shape[2]) // 2
        top = (t.shape[3] - shape[3]) // 2
        near = (t.shape[4] - shape[4]) // 2
        right = left + shape[2]
        bottom = top + shape[3]
        far = near + shape[4]
        assert left >= 0 and top >= 0 and near >= 0 and right < t.shape[2] and bottom < t.shape[3] and far < t.shape[4], \
            'Cropping image is less shape than input shape.\nInput shape:{}, Cropping shape:{}, (L,R,T,B):({},{},{},{})'.format(
                t.shape, shape, left, right, top, bottom)
        return t[:, :, left:right, top:bottom, near:far]
    raise NotImplementedError('Nd cropping is not inmplemented.')


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
                self.c = L.ConvolutionND(
                    n_dims, in_ch, out_ch, ksize=ksize, stride=stride, pad=1, initialW=w)
            elif sample == 'up':
                self.c = L.DeconvolutionND(
                    n_dims, in_ch, out_ch, ksize=ksize, stride=stride, pad=1, initialW=w)
            elif sample == 'up_shuffle':
                self.c = PixelShuffleUpsampler(
                    n_dims, in_ch, out_ch, 2,
                    ksize=ksize, stride=stride,
                )
            else:
                raise KeyError('Unknown sampling type:' + sample)
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


class ResBlock(chainer.Chain):
    dropout = dict(
        bayesian=bayesian_dropout,
        dropout=F.dropout,
        none=None,
    )

    def __init__(self,
                 n_dims, in_ch, lat_ch, out_ch,
                 ksize=3, stride=1, activation=F.relu,
                 dropout='dropout', method='post'
                 ):
        self.activation = activation
        w = chainer.initializers.Normal(0.02)

        if method == 'post':
            self.method = ResBlock.post_activation
        elif method == 'pre':
            self.method = ResBlock.pre_activation
        else:
            raise ValueError(
                'A method is either "pre" or "post". Actual: ' + method)
        self.dropout = ResBlock.dropout[dropout]

        super().__init__()
        with self.init_scope():
            self.c1 = L.ConvolutionND(
                n_dims, in_ch, lat_ch, ksize=ksize, stride=stride, pad=1, initialW=w)
            self.bn1 = L.BatchNormalization(lat_ch)
            self.c2 = L.ConvolutionND(
                n_dims, lat_ch, out_ch, ksize=ksize, stride=stride, pad=1, initialW=w)
            self.bn2 = L.BatchNormalization(out_ch)

    def __call__(self, x):
        return self.method(self, x)

    def post_activation(self, x):
        h = x
        h = self.c1(h)
        h = self.bn1(h)
        h = self.activation(h)

        h = self.c2(h)
        h = self.bn2(h)

        h = self.activation(h + x)

        if self.dropout:
            h = F.dropout(h)

        return h

    def pre_activation(self, x):
        h = x
        h = self.bn1(h)
        h = self.activation(h)
        h = self.c1(h)

        h = self.bn2(h)
        h = self.activation(h + x)

        if self.dropout:
            h = F.dropout(h)

        h = self.c2(x)

        return h + x


class PixelShuffleUpsampler(chainer.Chain):
    """Pixel Shuffler for the super resolution.
    This upsampler is effective upsampling method compared with the deconvolution.
        The deconvolution has a problem of the checkerboard artifact.
        A detail of this problem shows the following.
        http://distill.pub/2016/deconv-checkerboard/

    See also:
        https://arxiv.org/abs/1609.05158
    """

    def __init__(self,
                 n_dims, in_ch, out_ch, resolution=2,
                 ksize=3, stride=1
                 ):
        super().__init__()

        self.n_dims = n_dims
        self.resolution = resolution
        self.in_channels = in_ch
        self.out_channels = out_ch

        with self.init_scope():
            m = self.resolution ** self.n_dims
            self.conv = L.ConvolutionND(
                n_dims, in_ch, out_ch * m, ksize, stride
            )

    def __call__(self, x):
        r = self.resolution
        out = self.conv(x)
        batchsize = out.shape[0]
        in_channels = out.shape[1]
        out_channels = self.out_channels

        in_shape = out.shape[2:]
        out_shape = tuple(s * r for s in in_shape)

        r_tuple = tuple(self.resolution for _ in range(self.n_dims))
        out = F.reshape(out, (batchsize, out_channels,) + r_tuple + in_shape)
        out = F.transpose(out, self.make_transpose_indices())
        out = F.reshape(out, (batchsize, out_channels, ) + out_shape)
        return out

    def make_transpose_indices(self):
        si = [0, 1]
        si.extend([2 * (i + 1) + 1 for i in range(self.n_dims)])
        si.extend([2 * (i + 1) for i in range(self.n_dims)])
        return si
