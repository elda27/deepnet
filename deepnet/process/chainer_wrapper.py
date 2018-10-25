from chainer import functions as F
from deepnet.core.registration import add_process

add_process('chainer.reshape', F.reshape)
add_process('chainer.mean', F.mean)
add_process('chainer.sigmoid', F.sigmoid)
add_process('chainer.softmax', F.softmax)
add_process('chainer.argmax', F.argmax)
add_process('chainer.transpose', F.transpose)
add_process('chainer.expand_dims', F.expand_dims)
add_process('chainer.sigmoid_cross_entropy', F.sigmoid_cross_entropy)
add_process('chainer.softmax_cross_entropy', F.softmax_cross_entropy)
add_process('chainer.batch_l2_norm_squared', F.batch_l2_norm_squared)
