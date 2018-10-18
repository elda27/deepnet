import chainer
from deepnet.network import utils
from deepnet.network.conv_auto_encoder import scae as conv_auto_encoder
import chainer.functions as F
import chainer.links as L

from deepnet.core.registration import register_network
from corenet import declare_node_type


@register_network(
    'network.tl-net.ssegnet',
    wrap_args={'decoder': 'network'}
)
@declare_node_type('chainer')
class SSegnet(chainer.Chain):
    def __init__(self,
                 n_dim,
                 in_channel,
                 output_shape=None,
                 decoder=None,
                 use_skipping_connection=True
                 ):
        self.n_dim = n_dim
        self.output_shape = output_shape

        self.stores = dict()

        super().__init__()
        self.decoders = []
        self.use_skipping_connection = use_skipping_connection
        with self.init_scope():
            if isinstance(decoder, list):
                for i, d in enumerate(decoder):
                    decoder_name = 'decoder{}'.format(i)
                    setattr(self, decoder_name, d)
                    self.decoders.append(decoder_name)
                decoder = getattr(self, 'decoder0')
            else:
                self.decoder = decoder
                self.decoders.append(decoder)

        self.encoder = conv_auto_encoder.SEncoder(
            n_dim, in_channel,
            n_latent_elem=decoder.n_latent_elem,
            n_layers=decoder.n_layers,
            n_res_layers=decoder.n_res_layers,
            n_units=decoder.n_units,
            dropout=decoder.dropout,
            use_batch_norm=decoder.use_batch_norm,
            use_skip_connection=decoder.use_skip_connection
        )

    def __call__(self, x):
        h = self.encoder(x)
        self.stores['encoder'] = h

        if hasattr(self, 'decoder'):
            return self.decode('decode', x, h)
        else:
            decode_results = []
            for decoder_name in self.decoders:
                decode_results.append(self.decode(decoder_name, x, h))

            return decode_results

    def decode(self, decoder_name, x, h):
        decoder = getattr(self, decoder_name)

        h = decoder(h, connections=self.encoder.stores)

        output_shape = None
        if self.output_shape is None:
            output_shape = x.shape
        else:
            output_shape = (
                x.shape[0], decoder.out_channel) + tuple(self.output_shape)
        h = utils.crop(h, output_shape, decoder.n_dim)

        self.stores[decoder_name] = h
