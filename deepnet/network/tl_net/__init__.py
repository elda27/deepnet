import chainer
from deepnet.network import utils, conv_auto_encoder
import chainer.functions as F
import chainer.links as L

from deepnet.network.init import register_network

@register_network('network.tl-net.segnet')
class Segnet(chainer.Chain):
    def __init__(self,
        n_dim,
        in_channel, 
        decoder = None, 
        use_skipping_connection='none'
        ):
        super().__init__()
        self.use_skipping_connection = use_skipping_connection
        with self.init_scope():
            if decoder is not None:
                self.decoder = decoder
            self.encoder = conv_auto_encoder.Encoder(
                n_dim, in_channel,
                encode_dim= self.decoder.input_dim,
                n_layers= self.decoder.n_layers,
                n_units = self.decoder.n_units
                )
        
        self.layers = dict(
            decoder=self.decoder,
            encoder=self.encoder
        )
        self.stores = dict()

    def __call__(self, x):
        h = self.encoder(x)
        self.stores['encoder'] = h

        old_skip_flag = self.decoder.use_skipping_connection
        self.decoder.use_skipping_connection = self.use_skipping_connection

        h = self.decoder(h, connections=self.encoder.stores)
        h = utils.crop(h, x.shape)
        self.stores['decoder'] = h

        self.decoder.use_skipping_connection = old_skip_flag
        
        return h
