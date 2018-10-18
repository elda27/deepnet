import chainer
from deepnet.network import utils
import chainer.functions as F
import chainer.links as L
import math
import numpy as np
from functools import reduce

from deepnet.core.registration import register_network
from corenet import declare_node_type


@register_network('network.scae.encoder')
@declare_node_type('chainer')
class SEncoder(chainer.Chain):
    def __init__(self,
                 n_dim, in_channel,
                 n_latent_elem=64, n_layers=4,
                 n_units=32, n_conv_layers=2, n_res_layers=0,
                 dropout='none', use_batch_norm=True,
                 use_skip_connection=False,
                 keep_latent_shape=True,
                 store_params=[],
                 ):
        self.n_dim = n_dim
        self.n_channel = in_channel
        self.n_latent_elem = n_latent_elem
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_conv_layers = n_conv_layers
        self.n_res_layers = n_res_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.use_skip_connection = use_skip_connection
        self.keep_latent_shape = keep_latent_shape
        self.store_params = store_params

        self.layers = {}
        self.stores = {}

        n_units = self.n_channel
        n_next_units = self.n_units
        for i in range(self.n_layers):
            self.layers['c{}_{}'.format(i, 0)] = utils.CBR(
                n_dim, n_units, n_next_units,
                ksize=3, stride=1, bn=use_batch_norm, sample='down',
                activation=F.relu, dropout=dropout,
            )

            for j in range(1, self.n_conv_layers):
                self.layers['c{}_{}'.format(i, j)] = utils.CBR(
                    n_dim, n_next_units, n_next_units,
                    ksize=3, stride=1, bn=use_batch_norm, sample='down',
                    activation=F.relu, dropout=dropout,
                )

            if self.use_skip_connection:
                self.store_params.append(
                    'c{}_{}'.format(i, self.n_conv_layers - 1)
                )

            self.layers['d{}'.format(i)] = utils.CBR(
                n_dim, n_next_units, n_next_units,
                ksize=4, stride=2, bn=use_batch_norm, sample='down',
                activation=F.relu, dropout=dropout,
            )

            n_units = n_next_units
            n_next_units = n_next_units * 2

        for i in range(self.n_res_layers):
            self.layers['res{}'.format(i)] = utils.ResBlock(
                n_dim, n_units, n_units, n_units
            )

        self.layers['flatten'] = utils.CBR(
            n_dim, n_units, 1,
            ksize=1, stride=1, bn=use_batch_norm, sample='down',
            activation=F.relu, dropout=dropout
        )

        if not self.keep_latent_shape:
            self.layers['fc'] = L.Linear(None, out_size=self.n_latent_elem)

        super().__init__(**self.layers)

    def __call__(self, x):
        h = x
        for i in range(0, self.n_layers):
            for j in range(0, self.n_conv_layers):
                h = self.apply('c{}_{}'.format(i, j), h)
            h = self.apply('d{}'.format(i), h)

        for i in range(self.n_res_layers):
            h = self.apply('res{}'.format(i), h)

        h = self.apply('flatten', h)
        if not self.keep_latent_shape:
            h = self.apply('fc', h)
        return h

    def apply(self, layer_name, x):
        x = self.layers[layer_name](x)
        if layer_name in self.store_params:
            self.stores[layer_name] = x
        return x


@register_network('network.scae.decoder')
@declare_node_type('chainer')
class SDecoder(chainer.Chain):
    def __init__(self,
                 n_dim, out_channel,
                 n_latent_elem=64,
                 upsample_start_shape=None,
                 n_layers=4, n_units=32,
                 n_conv_layers=2, n_res_layers=0,
                 dropout='none', use_batch_norm=True,
                 use_skip_connection=False,
                 upsampler='up_shuffle',
                 keep_latent_shape=True,
                 store_params=[],
                 ):
        self.n_dim = n_dim
        self.n_channel = out_channel
        self.n_latent_elem = n_latent_elem
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_conv_layers = n_conv_layers
        self.n_res_layers = n_res_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.use_skip_connection = use_skip_connection
        self.upsampler = upsampler
        self.keep_latent_shape = keep_latent_shape
        self.store_params = store_params

        if upsample_start_shape is None:
            self.upsample_start_shape = (8,) * n_dim
        else:
            self.upsample_start_shape = upsample_start_shape

        self.layers = {}
        self.stores = {}
        self.combine_layers = []

        n_units = self.n_channel
        n_last_units = self.n_units
        for i in range(self.n_layers):
            self.layers['c{}_{}'.format(i, 0)] = utils.CBR(
                n_dim, n_last_units, n_units,
                ksize=3, stride=1, bn=use_batch_norm, sample='down',
                activation=F.relu, dropout=dropout,
            )

            if self.use_skip_connection:
                n_conv_layers = self.n_conv_layers - 1
            else:
                n_conv_layers = self.n_conv_layers

            for j in range(1, n_conv_layers):
                self.layers['c{}_{}'.format(i, j)] = utils.CBR(
                    n_dim, n_last_units, n_last_units,
                    ksize=3, stride=1, bn=use_batch_norm, sample='down',
                    activation=F.relu, dropout=dropout,
                )

            if self.use_skip_connection:
                # Convolution and combined with before downsampled image.
                layer_name = 'c{}_{}'.format(i, n_conv_layers)
                self.layers[layer_name] = utils.CBR(
                    n_dim, n_last_units * 2, n_last_units,
                    ksize=3, stride=1, bn=use_batch_norm, sample='down',
                    activation=F.relu, dropout=dropout,
                )
                self.combine_layers.append(layer_name)

            self.layers['d{}'.format(i)] = utils.CBR(
                n_dim, n_units, n_units,
                ksize=4, stride=2, bn=use_batch_norm, sample=self.upsampler,
                activation=F.relu, dropout=dropout,
            )

            n_units = n_last_units
            n_last_units = n_last_units * 2

        for i in range(self.n_res_layers):
            self.layers['res{}'.format(i)] = utils.ResBlock(
                n_dim, n_units, n_units, n_units
            )

        self.flatten_units = n_units
        self.layers['reconstruct'] = utils.CBR(
            n_dim, 1, n_units,
            ksize=4, stride=2, bn=use_batch_norm, sample=self.upsampler,
            activation=F.relu, dropout=dropout
        )

        if not self.keep_latent_shape:
            self.layers['fc'] = L.Linear(self.n_latent_elem)

        super().__init__(**self.layers)

    def __call__(self, x, stores=None):
        h = x
        if not self.keep_latent_shape:
            h = self.apply('fc', h)
            h = F.reshape(
                h, (h.shape[0], self.flatten_units,) +
                self.upsample_start_shape
            )

        h = self.apply('reconstruct', h)

        for i in reversed(range(self.n_res_layers)):
            h = self.apply('res{}'.format(i), h)

        for i in reversed(range(0, self.n_layers)):
            for j in reversed(range(0, self.n_conv_layers)):
                h = self.apply('c{}_{}'.format(i, j), h, stores)
            h = self.apply('d{}'.format(i), h)

        return h

    def apply(self, layer_name, h, stores=None):
        if self.use_skip_connection and layer_name in self.combine_layers:
            assert stores is not None

            source = stores[layer_name]

            assert h.shape[1] == source.shape[1], \
                'Unmatched num units\nDecoding unit:{}, Encoded unit:{}'.format(
                    h.shape, source.shape)

            if all([hs > fhs for hs, fhs in zip(h.shape[2:], source.shape[2:])]):
                # Decoding image is larger than encoder image
                padding_pix = (
                    np.array(h.shape[2:]) - np.array(source.shape[2:])
                ) / 2
                pads = [
                    (0, 0), (0, 0)] + [(int(math.floor(pix)),
                                        int(math.ceil(pix))) for pix in padding_pix]
                source = F.pad(source, pads, 'constant')
            else:
                source = utils.crop(source, h.shape, self.n_dim)

            h = F.concat((h, source), axis=1)

        h = self.layers[layer_name](h)

        if layer_name in self.store_params:
            self.stores[layer_name] = h
        return h


@register_network('network.scae')
@declare_node_type('chainer')
class ConvolutionalAutoEncoder(chainer.Chain):
    def __init__(self,
                 n_dim, in_channel, out_channel,
                 n_latent_elem=64, n_layers=4,
                 n_res_layers=0,
                 n_tasks=1,
                 dropout='none', upsampler='up_shuffle',
                 use_batch_norm=True,
                 use_skip_connection=True,
                 keep_latent_shape=True,
                 encoder_store_params=[],
                 decoder_store_params=[],
                 ):

        self.layers = {}
        self.stores = {}
        self.n_dim = n_dim
        self.n_tasks = n_tasks
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.use_skip_connection = use_skip_connection
        self.in_channel = in_channel
        if isinstance(out_channel, list):
            self.out_channel = out_channel
        else:
            self.out_channel = [out_channel]

        self.decoders = []

        super().__init__()
        with self.init_scope():
            self.encoder = SEncoder(
                n_dim, in_channel,
                n_latent_elem=n_latent_elem,
                n_layers=n_layers, n_res_layers=n_res_layers,
                dropout=dropout, use_batch_norm=use_batch_norm,
                use_skip_connection=use_skip_connection,
                keep_latent_shape=keep_latent_shape,
                store_params=encoder_store_params
            )
            for i in range(self.n_tasks):
                setattr(self, 'decoder{}'.format(i), SDecoder(
                    n_dim, self.out_channel[i],
                    n_latent_elem=n_latent_elem,
                    n_layers=n_layers,
                    n_units=32, n_res_layers=n_res_layers,
                    upsampler=upsampler, dropout=dropout,
                    use_batch_norm=use_batch_norm,
                    use_skip_connection=use_skip_connection,
                    keep_latent_shape=keep_latent_shape,
                    store_params=decoder_store_params
                ))
                self.decoders.append(getattr(self, 'decoder{}'.format(i)))

        self.layers['encoder'] = self.encoder
        for i, decoder in enumerate(self.decoders):
            self.layers['decoder{}'.format(i)] = decoder

    def __call__(self, x):
        h = self.encoder(x)

        self.stores['encoder'] = h

        results = []
        latent_h = h
        for decoder in self.decoders:
            if self.use_skip_connection:
                h = decoder(latent_h, self.encoder.stores)
            else:
                h = decoder(h, self.encoder.stores)
            h = utils.crop(h, x.shape, self.n_dim)
            results.append(h)

        return results
