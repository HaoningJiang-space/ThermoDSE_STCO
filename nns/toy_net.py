from core.layer import *
from core.network import Network

NN = Network('toyNet')

NN.set_input_layer(InputLayer(nofm=512, sofm=14))

NN.add('conv1', ConvLayer(512, 1024, 14, 3, 1), ifm_prevs=(NN.INPUT_LAYER_KEY,))
NN.add('conv2', ConvLayer(1024, 2048, 14, 1, 1), ifm_prevs=('conv1',))
# NN.add('pool1', PoolingLayer(64, 56, 3, 2), prevs=('conv1',))
