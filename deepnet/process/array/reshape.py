from corenet import declare_node_type
from deepnet.core.registration import register_process
from deepnet.core import config
import chainer
import chainer.functions as F


@register_process()
def batch_flatten(*input):
    output = []
    for i in input:
        output.append(F.reshape(i, (i.shape[0], sum(i.shape[1:]))))
    return output
