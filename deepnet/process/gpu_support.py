from deepnet.core.registration import register_process

NO_CUPY = False
try:
    import cupy
except ImportError:
    NO_CUPY = True

import chainer
from chainer import cuda, functions as F

@register_process()
def to_cpu(*input_list):
    output_list = []
    for input_ in input_list:
        if isinstance(input_, list):
            input_ = [ F.expand_dims(chainer.Variable(i), axis=0) for i in input_ ]
            input_ = F.concat(input_, axis=0)
        else:
            input_ = chainer.Variable(input_)
        output_list.append(input_)
    return output_list

@register_process()
def to_gpu(*input_list):
    if NO_CUPY:
        return to_cpu(*input_list)
        
    output_list = []
    for input_ in input_list:
        if isinstance(input_, list):
            #input_ = [ F.expand_dims(chainer.Variable(i), axis=0) for i in input_ ]
            #input_ = F.concat(input_, axis=0)
            input_ = chainer.Variable(np.concatenate([ np.expand_dims(i.astype(np.float32), axis=0) for i in input_ ], axis=0))
        elif isinstance(input_, chainer.Variable):
            input_ = F.copy(input_, -1)
        else:
            input_ = chainer.Variable(input_.astype(np.float32))
        input_.to_gpu()
        output_list.append(input_)
    return output_list