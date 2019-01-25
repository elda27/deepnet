import chainer
from deepnet.network import utils, conv_auto_encoder
import chainer.functions as F
import chainer.links as L

from deepnet.core.registration import register_network
from corenet import declare_node_type

from . import ssegnet


@register_network(
    'network.tl-net.segnet',
    wrap_args={'decoder': 'network'}
)
@declare_node_type('chainer')
class Segnet(chainer.Chain):
    def __init__(self,
                 n_dim,
                 in_channel,
                 output_shape=None,
                 decoder=None,
                 use_skipping_connection='none'
                 ):
        self.n_dim = n_dim
        self.output_shape = output_shape

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
            self.encoder = conv_auto_encoder.Encoder(
                n_dim, in_channel,
                encode_dim=decoder.input_dim,
                n_layers=decoder.n_layers,
                n_res_layers=decoder.n_res_layers,
                n_units=decoder.n_units,
                dropout=decoder.dropout,
                use_batch_norm=decoder.use_batch_norm
            )

        self.layers = dict(
            decoder=self.decoder,
            encoder=self.encoder
        )
        self.stores = dict()

    def __call__(self, x):
        h = self.encoder(x)
        self.stores['encoder'] = h
        self.stores['fc'] = h
        self.decode_latent_vector(h, x=x)

    def decode_latent_vector(self, x, h):
        if hasattr(self, 'decoder'):
            return self.decode('decoder', x, h)
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
        return h
