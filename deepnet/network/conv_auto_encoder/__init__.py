import chainer
from deepnet.network import utils
import chainer.functions as F
import chainer.links as L
import math
import numpy as np
from functools import reduce

from deepnet.network.init import register_network

@register_network('network.cae.encoder')
class Encoder(chainer.Chain):
    def __init__(self,
        n_dim, in_channel, 
        encode_dim=64, n_layers=4, 
        n_units=32, 
        dropout='none', use_batch_norm=False, 
        #**kwargs
        ):
        self.layers = {}
        self.stores = {}
        self.n_layers = n_layers
        self.encode_dim = encode_dim
        
        w = chainer.initializers.Normal(0.02)

        # Layer declaration
        self.layers['c01'] = utils.CBR(n_dim, in_channel, n_units, ksize=3, stride=2, bn=use_batch_norm, sample='down', activation=F.relu, dropout=dropout)
        self.layers['c02'] = utils.CBR(n_dim, n_units, n_units, ksize=3, stride=1, bn=use_batch_norm, sample='down', activation=F.relu, dropout=dropout)

        n_unit = n_units
        for i in range(1, n_layers):
            next_unit = n_units * 2 ** i
            self.layers['c{}1'.format(i)] = utils.CBR(n_dim, n_unit, next_unit, ksize=3, stride=2, bn=use_batch_norm, sample='down', activation=F.relu, dropout=dropout)
            self.layers['c{}2'.format(i)] = utils.CBR(n_dim, next_unit, next_unit, ksize=3, stride=1, bn=use_batch_norm, sample='down', activation=F.relu, dropout=dropout)
            n_unit = next_unit
        
        self.n_end_units = n_unit

        self.layers['c{}1'.format(n_layers)] = utils.CBR(n_dim, n_unit, 1, stride=3, bn=use_batch_norm, sample='down', activation=F.relu, dropout=dropout)

        self.layers['fc'] = L.Linear(None, out_size=encode_dim)

        chainer.Chain.__init__(self, **self.layers)

    def __call__(self, x):
        h = x
        for i in range(0, self.n_layers):
            h = self.layers['c{}1'.format(i)](h)
            self.stores['c{}1'.format(i)] = h
            h = self.layers['c{}2'.format(i)](h)
            self.stores['c{}2'.format(i)] = h
        
        h = self.layers['c{}1'.format(self.n_layers)](h)
        self.stores['c{}1'.format(self.n_layers)] = h
        h = self.layers['fc'](h)
        self.stores['fc'] = h
        return h

@register_network('network.cae.decoder')
class Decoder(chainer.Chain):
    def __init__(
        self, n_dim, out_channel, 
        input_dim=64, upsample_start_shape = (8, 8), 
        n_layers=4, n_units=32, 
        dropout='none', use_batch_norm=False,
        use_skipping_connection = 'none',
        **kwargs
        ):
        assert use_skipping_connection in DECODER_APPLY_NEXT_DICT

        self.layers = {}
        self.stores = {}
        self.n_dim = n_dim
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.out_channel = out_channel
        self.n_units = n_units
        self.upsample_start_shape = upsample_start_shape
        self.use_skipping_connection = use_skipping_connection
        self.apply_next = DECODER_APPLY_NEXT_DICT[self.use_skipping_connection]

        # Layer declaration
        n_unit = n_units * 2 ** (self.n_layers - 1)
        self.eur = 2 if self.use_skipping_connection == 'concat' else 1
        self.layers['fc'] = L.Linear(input_dim, reduce(lambda x, y: x * y, upsample_start_shape) * n_unit)
        self.layers['c{}1'.format(n_layers)] = utils.CBR(n_dim, self.eur * n_unit, n_unit, ksize=7, stride=3, bn=use_batch_norm, sample='up', activation=F.relu, dropout=dropout)
        self.layers['c{}2'.format(n_layers)] = utils.CBR(n_dim, n_unit, n_unit // self.eur, ksize=3, stride=1, bn=use_batch_norm, sample='down', activation=F.relu, dropout=dropout)
        self.n_start_units = n_unit

        for i in range(n_layers - 1, 1, -1):
            next_unit = n_units * 2 ** (i - 1)
            self.layers['c{}1'.format(i)] = utils.CBR(n_dim, n_unit, next_unit, ksize=4, stride=2, bn=use_batch_norm, sample='up', activation=F.relu, dropout=dropout)
            self.layers['c{}2'.format(i)] = utils.CBR(n_dim, next_unit, next_unit // self.eur, ksize=3, stride=1, bn=use_batch_norm, sample='down', activation=F.relu, dropout=dropout)
            n_unit = next_unit 

        self.layers['c11'] = utils.CBR(n_dim, n_unit, n_unit, stride=2, bn=use_batch_norm, sample='up', activation=F.relu, dropout=dropout)
        self.layers['c12'] = L.ConvolutionND(n_dim, n_unit, out_channel, ksize=3, stride=1)

        chainer.Chain.__init__(self, **self.layers)

    def __call__(self, x, connections = None):
        h = F.relu(self.layers['fc'](x))
        self.stores['fc'] = h
        h = F.reshape(h, (h.shape[0], self.n_start_units,) + self.upsample_start_shape)
        for i in range(self.n_layers, 1, -1):
            h = self.apply_next(self, h, 'c{}1'.format(i), connections, from_name='c{}2'.format(i - 1))
            h = self.apply_next(self, h, 'c{}2'.format(i))
        h = self.apply_next(self, h, 'c11', connections, from_name='c02')
        h = self.apply_next(self, h, 'c12')
        return h

    def apply_next_default(self, h, name, connections=None, from_name=''):
        h = self.layers[name](h)
        self.stores[name] = h
        return h

    def apply_next_add(self, h, name, connections = None, from_name = ''):
        assert connections is not None
        assert from_name in connections
        from_h = connections[from_name]

        assert h.shape[1] == from_h.shape[1], 'Unmatched num units\nDecoding unit:{}, Encoded unit:{}'.format(h.shape, from_h.shape)

        if all( [hs > fhs for hs, fhs in zip(h.shape[2:], from_h.shape[2:])] ): # Decoding image is larger than encoder image
            padding_pix = (np.array(h.shape[2:]) - np.array(from_h.shape[2:])) / 2
            pads = [(0, 0), (0, 0)] + [ (int(math.floor(pix)), int(math.ceil(pix))) for pix in padding_pix ]
            from_h = F.pad(from_h, pads, 'constant')
        else:
            from_h = utils.crop(from_h, h.shape)

        h = h + from_h
        h = self.layers[name](h)
        self.stores[name] = h
        return h

    def apply_next_concat(self, h, name, connections = None, from_name = ''):
        assert connections is not None
        assert from_name in connections
        from_h = connections[from_name]

        assert h.shape[1] == from_h.shape[1], 'Unmatched num units\nDecoding unit:{}, Encoded unit:{}'.format(h.shape, from_h.shape)

        if all( [hs > fhs for hs, fhs in zip(h.shape[2:], from_h.shape[2:])] ): # Decoding image is larger than encoder image
            padding_pix = (np.array(h.shape[2:]) - np.array(from_h.shape[2:])) / 2
            pads = [(0, 0), (0, 0)] + [ (int(math.floor(pix)), int(math.ceil(pix))) for pix in padding_pix ]
            from_h = F.pad(from_h, pads, 'constant')
        else:
            from_h = utils.crop(from_h, h.shape)

        h = F.concat((h, from_h), axis=1)
        h = self.layers[name](h)
        self.stores[name] = h
        return h
    
DECODER_APPLY_NEXT_DICT = {
    'none': Decoder.apply_next_default,
    'add': Decoder.apply_next_add,
    'concat': Decoder.apply_next_concat,
}

def ValiationAutoEncoderUnit(input_vector):
    vector_dim = input_vector.shape[1]
    representation_dim = int(vector_dim // 2)

    xp = input_vector.xp

    ones = xp.ones((representation_dim, representation_dim), dtype=xp.float32)
    ones = chainer.Variable(ones)
    return input_vector[:representation_dim] + input_vector[representation_dim:] * F.gaussian(0, ones)

@register_network('network.cae')
class ConvolutionalAutoEncoder(chainer.Chain):
    def __init__(self, 
        n_dim, in_out_channel, 
        encode_dim=64, n_layers=4, 
        dropout='none', use_batch_norm=False,
        use_skipping_connection='none', 
        vae_unit=None,
        **kwargs):
        self.layers = {}
        self.n_dim = n_dim
        self.use_skipping_connection = use_skipping_connection
        self.vae_unit = None
        chainer.Chain.__init__(self)
        with self.init_scope():
            self.encoder = Encoder(
                n_dim, in_out_channel, 
                encode_dim=encode_dim, 
                n_layers=n_layers, 
                dropout=dropout, use_batch_norm=use_batch_norm
                )
            self.decoder = Decoder(
                n_dim, in_out_channel, 
                input_dim=encode_dim, 
                n_layers=n_layers, 
                n_start_units=32,
                dropout=dropout, use_batch_norm=use_batch_norm,
                use_skipping_connection=use_skipping_connection
                )
        self.layers['encoder'] = self.encoder
        self.layers['decoder'] = self.decoder

    def __call__(self, x):
        h = self.encoder(x)

        if self.use_skipping_connection:
            h = self.decoder(h, self.encoder.stores)
        else:
            h = self.decoder(h)
        h = utils.crop(h, x.shape)
        return h
