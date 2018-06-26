import chainer
from deepnet.network import utils
import chainer.functions as F
import chainer.links as L

class Descriminator(chainer.Chain):
    def __init__(self, real_ch, fake_ch, n_layers=4, n_units=32, **kwargs):
        self.layers = {}
        w = chainer.initializers.Normal(0.02)
        self.layers['c0_0'] = utils.CBR(real_ch, n_units, bn=False, sample='down', activation=F.leaky_relu, dropout=None)
        self.layers['c0_1'] = utils.CBR(fake_ch, n_units, bn=False, sample='down', activation=F.leaky_relu, dropout=None)

        for i in range(1, n_layers):
            n_unit = n_units * 2 ** i
            self.layers['c' + str(i)] = utils.CBR(n_unit, n_unit * 2, bn=True, sample='down', activation=F.leaky_relu, dropout=None)
        self.layers['c' + str(n_layers)] = L.Convolution2D(n_units * 2 ** n_layers, 1, ksize=3, initialW=w)

        chainer.Chain.__init__(**self.layers)


    def __call__(self, x_0, x_1):
        h = F.concat([self.c0_0(x_0), self.c0_1(x_1)])
        for i in range(1, self.n_layers + 1):
            h = self.layers['c' + str(i)](h)
        return h