import chainer
from deepnet.network import utils, conv_auto_encoder
import chainer.functions as F
import chainer.links as L

from deepnet.core.registration  import register_network
from corenet import declare_node_type

@register_network(
    'network.tl-net.segnet',
    wrap_args={ 'decoder':'network' }
    )
@declare_node_type('chainer')
class Segnet(chainer.Chain):
    def __init__(self,
        n_dim,
        in_channel, 
        output_shape = None,
        decoder = None, 
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
            else:
                self.decoder = decoder
                self.decoders.append(decoder)
            self.encoder = conv_auto_encoder.Encoder(
                n_dim, in_channel,
                encode_dim= self.decoder.input_dim,
                n_layers= self.decoder.n_layers,
                n_res_layers= self.decoder.n_res_layers,
                n_units = self.decoder.n_units,
                dropout = self.decoder.dropout,
                use_batch_norm= self.decoder.use_batch_norm
                )
        
        self.layers = dict(
            decoder=self.decoder,
            encoder=self.encoder
        )
        self.stores = dict()

    def __call__(self, x):
        h = self.encoder(x)
        self.stores['encoder'] = h
        
        if hasattr(self, 'decoder'):
            return self.decode('decode', h)
        else:
            decode_results = []
            for decoder_name in self.decoders:
                decode_results.append(decode(decoder_name, h))

            return decode_results

    def decode(self, deocder_name, h):
        decoder = getattr(self, deocder_name)
        old_skip_flag = decoder.use_skipping_connection
        decoder.use_skipping_connection = self.use_skipping_connection

        h = decoder(h, connections=self.encoder.stores)

        output_shape = None
        if self.output_shape is None:
            output_shape = x.shape
        else:
            output_shape = (x.shape[0], decoder.out_channel) + tuple(self.output_shape)
        h = utils.crop(h, output_shape, decoder.n_dim)

        self.stores[decoder_name] = h

        decoder.use_skipping_connection = old_skip_flag
        