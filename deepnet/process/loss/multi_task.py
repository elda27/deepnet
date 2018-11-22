from deepnet.core.registration import register_network
from corenet import declare_node_type
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers
from chainer.initializers import Constant
from chainer.backends import cuda
import numpy as np
import functools


@register_network('network.multi_task_loss')
@declare_node_type('chainer')
class MultiTaskLoss(chainer.Chain):
    def __init__(self, loss_types, initialize=None):
        super().__init__()
        self.loss_types = loss_types
        self.combine_methods = []

        if initialize is None:
            initialize = [Constant(1.0)
                          for i in range(len(loss_types))]
        elif isinstance(initialize, (list, tuple)):
            if isinstance(initialize[0], float):
                initialize = [Constant(i) for i in initialize]
        else:
            initialize = [initialize for _ in loss_types]

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

    def __call__(self, *losses):
        loss = None
        #w_root = 1.0
        for i, combine_method in enumerate(self.combine_methods):
            W = getattr(self, 'sigma_{}'.format(i))
            W = F.broadcast_to(W, losses[i].shape)
            if loss is None:
                loss = F.sum(combine_method(losses[i], W))
            else:
                loss += F.sum(combine_method(losses[i], W))
        return loss

    def combine_softmax_cross_entropy(self, loss, W):
        return F.exp(-2.0 * W) * loss + W

    def combine_euclidean(self, loss, W):
        return F.exp(-W) * loss + W
