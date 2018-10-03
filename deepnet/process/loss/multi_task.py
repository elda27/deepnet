from deepnet.core.registration import register_network
from corenet import declare_node_type
import chainer
import chainer.functions as F
import chainer.initializers
import numpy as np
import functools


@register_network('network.multi_task_loss')
@declare_node_type('chainer')
class MultiTaskLoss(chainer.Chain):
    def __init__(self, loss_types, initialize=None, fixed=False):
        super().__init__()
        self.loss_types = loss_types
        self.combine_methods = []
        self.fixed = fixed

        if initialize is None:
            initialize = [chainer.initializers.Constant(1.0)
                          for i in range(len(loss_types))]

        for i, loss_type in enumerate(self.loss_types):
            with self.init_scope():
                w = initialize[i]
                setattr(
                    self, 'sigma_{}'.format(i),
                    chainer.Parameter(
                        initializer=w,
                        shape=(1,)
                    )
                )

            if loss_type == 'softmax_cross_entropy':
                self.combine_methods.append(self.combine_softmax_cross_entropy)
            elif loss_type == 'euclidean':
                self.combine_methods.append(self.combine_euclidean)
            else:
                raise AttributeError()

        if fixed:
            self.disable_update()

    def __call__(self, *losses):
        loss = None
        #w_root = 1.0
        for i, combine_method in enumerate(self.combine_methods):
            W = getattr(self, 'sigma_{}'.format(i))
            #W = W * w_root
            if loss is None:
                loss = combine_method(
                    F.reshape(losses[i], (1,)), F.reshape(W, (1,)))
            else:
                loss += combine_method(
                    F.reshape(losses[i], (1,)), F.reshape(W, (1,)))
        return loss

    def log(self, W):
        xp = chainer.cuda.get_array_module(W)
        f_min = chainer.Variable(xp.ones_like(W) * 1e-8)
        return F.log(F.maximum(W, f_min))

    def combine_softmax_cross_entropy(self, loss, W):
        return 1.0 / (W ** 2) * loss + self.log(W)

    def combine_euclidean(self, loss, W):
        return 1.0 / (2.0 * W ** 2) * loss + self.log(W)
