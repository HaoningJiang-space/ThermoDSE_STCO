from core.layer import *
from core.network import Network

NN = Network('yolov2')

NN.set_input_layer(InputLayer(3, 224))

NN.add('conv1', ConvLayer(3,32,224,3))
NN.add('pool1', PoolingLayer(32,112,2,2))

NN.add('conv2', ConvLayer(32,64,112,3))
NN.add('pool2', PoolingLayer(64,56,2,2))

NN.add('conv3', ConvLayer(64,128,56,3))
NN.add('conv4', ConvLayer(128,64,56,1))
NN.add('conv5', ConvLayer(64,128,56,3))
NN.add('pool3', PoolingLayer(128,28,2,2))

NN.add('conv6', ConvLayer(128,256,28,3))
NN.add('conv7', ConvLayer(256,128,28,1))
NN.add('conv8', ConvLayer(128,256,28,3))
NN.add('pool4', PoolingLayer(256,14,2,2))

NN.add('conv9 ', ConvLayer(256,512,14,3))
NN.add('conv10', ConvLayer(512,256,14,1))
NN.add('conv11', ConvLayer(256,512,14,3))
NN.add('conv12', ConvLayer(512,256,14,1))
NN.add('conv13', ConvLayer(256,512,14,3))
NN.add('pool5 ', PoolingLayer(512,7,2,2))


NN.add('conv14', ConvLayer(512,1024,7,3))
NN.add('conv15', ConvLayer(1024,512,7,1))
NN.add('conv16', ConvLayer(512,1024,7,3))
NN.add('conv17', ConvLayer(1024,512,7,1))
NN.add('conv18', ConvLayer(512,1024,7,3))

NN.add('pool6 ', PoolingLayer(1024,1,7, 7))