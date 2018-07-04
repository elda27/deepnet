import chainer
from deepnet.network import utils
import chainer.functions as F
import chainer.links as L
import math
import numpy as np
from functools import reduce

from deepnet.network.init import register_network

@register_network('network.alexnet')
class AlexNet(chainer.Chain):
    def __init__(self,
        n_dim, in_channel, 
        n_class, n_layers = 3, n_units = 64,
        dropout='none', use_batch_norm=False
        ):
        self.layers = {}
        self.stores = {}
        self.n_layers = n_layers
        self.n_class = n_class
        
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