import chainer
from deepnet.network import utils
import chainer.functions as F
import chainer.links as L
import math
import numpy as np
from deepnet.network.utils import bayesian_dropout
from functools import reduce

from deepnet.core.registration import register_network
from corenet import declare_node_type


@register_network('network.ae.encoder')
@declare_node_type('chainer')
class Encoder(chainer.Chain):
    dropouts = {
        'none': lambda x: x,
        'dropout': F.dropout,
        'bayesian_dropout': bayesian_dropout,
    }

    def __init__(self,
                 encode_dim=64, n_layers=16,
                 dropout='none', use_batch_norm=True,
                 ):
        self.layers = {}
        self.stores = {}
        self.n_layers = n_layers
        self.encode_dim = encode_dim
        self.use_batch_norm = use_batch_norm
        self.dropout = self.dropouts[dropout]

        # Layer declaration
        for i in range(self.n_layers):
            units = encode_dim * 2 ** (self.n_layers - i + 1)
            self.layers['fc' + str(i)] = L.Linear(None, units)
            if self.use_batch_norm:
                self.layers['bn' + str(i)] = L.BatchNormalization(units)

        self.layers['fc' + str(self.n_layers)] = \
            L.Linear(None, out_size=encode_dim)

        super().__init__(**self.layers)

    def __call__(self, x):
        h = x
        for i in range(self.n_layers):
            h = self.apply(h, i)
        h = self.layers['fc' + str(self.n_layers)](h)
        return h

    def apply(self, h, index):
        h = self.layers['fc' + str(index)](h)
        if self.use_batch_norm:
            h = self.layers['bn' + str(index)](h)
        h = self.dropout(h)
        h = F.relu(h)
        return h


@register_network('network.ae.decoder')
@declare_node_type('chainer')
class Decoder(chainer.Chain):
    dropouts = {
        'none': lambda x: x,
        'dropout': F.dropout,
        'bayesian_dropout': bayesian_dropout,
    }

    def __init__(self,
                 output_dim,
                 encode_dim=64, n_layers=16,
                 dropout='none', use_batch_norm=False
                 ):

        self.layers = {}
        self.n_layers = n_layers
        self.n_layers = n_layers
        self.encode_dim = encode_dim
        self.use_batch_norm = use_batch_norm
        self.dropout = self.dropouts[dropout]

        # Layer declaration
        for i in range(self.n_layers, 0, -1):
            units = encode_dim * 2 ** (self.n_layers - i + 1)
            self.layers['fc' + str(i)] = L.Linear(None, units)
            if self.use_batch_norm:
                self.layers['bn' + str(i)] = L.BatchNormalization(units)

        self.layers['fc0'] = L.Linear(None, output_dim)

        super().__init__(**self.layers)

    def __call__(self, x):
        h = F.reshape(x, (x.shape[0], -1))
        for i in range(self.n_layers + 1, 0):
            h = self.apply(h, i)
        h = self.layers['fc0'](h)
        return h

    def apply(self, h, index):
        h = self.layers['fc' + str(index)](h)
        if self.use_batch_norm:
            h = self.layers['bn' + str(index)](h)
        h = self.dropout(h)
        h = F.relu(h)
        return h


@register_network('network.ae')
@declare_node_type('chainer')
class AutoEncoder(chainer.Chain):
    def __init__(self,
                 output_dim,
                 encode_dim=64, n_layers=16,
                 dropout='none', use_batch_norm=True,
                 ):

        self.layers = {}
        self.stores = {}
        self.dropout = dropout
        self.encode_dim = encode_dim
        self.n_layers = n_layers

        chainer.Chain.__init__(self)
        with self.init_scope():
            self.encoder = Encoder(
                encode_dim=encode_dim, n_layers=n_layers,
                dropout=dropout, use_batch_norm=use_batch_norm,
            )
            self.decoder = Decoder(
                output_dim,
                encode_dim=encode_dim, n_layers=n_layers,
                dropout=dropout, use_batch_norm=use_batch_norm
            )

        self.layers['encoder'] = self.encoder
        self.layers['decoder'] = self.decoder

    def __call__(self, x):
        h = self.encoder(x)

        # if self.latent_activation:
        #    h = F.sigmoid(h)

        self.stores['encoder'] = h

        results = []
        latent_h = h
        # for decoder in self.decoders:
        #    if self.use_skipping_connection:
        #        h = decoder(latent_h, self.encoder.stores)
        #    else:
        #        h = decoder(h)
        #    h = utils.crop(h, x.shape, self.n_dim)
        #    results.append(h)
        h = self.decoder(h)

        return h
