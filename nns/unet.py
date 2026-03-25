from core.layer import *
from core.network import Network

def add_douleConv(network:Network, name:str, nifm, nofm, sofm, prevs=None, mid_ofm = None):
    if not mid_ofm:
            mid_channels = nofm
    network.add(name+'_conv1', ConvLayer(nifm=nifm,nofm=mid_channels, sofm=sofm,sfil=3), ifm_prevs=prevs)
    network.add(name+'_conv2', ConvLayer(nifm=mid_channels,nofm=nofm, sofm=sofm,sfil=3))
    return name+'_conv2'

def add_down(network:Network, name:str, nifm, nofm, sofm, prevs):
    pool_name = name + '_pool'
    network.add(pool_name, PoolingLayer(nifm, sofm,2), ifm_prevs= prevs)
    down_name = add_douleConv(network, name+'_dconv' ,nifm = nifm, nofm=nofm, sofm=sofm,prevs=pool_name)
    return down_name

def add_up(network:Network, name:str, nifm, nofm, sofm, prevs):
    assert len(prevs) == 2  # since the 
    x1 = prevs[0]
    x2 = prevs[1]
    up_name = name + '_up'
    network.add(up_name, UpsampleLayer(nofm=nifm // 2, sofm=sofm, scale=2),ifm_prevs=x1)
    conv_name1 = name + '_gpconv1' # concate and convolution
    network.add(conv_name1, groupConvLayer(nifm=nifm, nofm=nofm, sofm=sofm,sfil=3, numG=2, concat_hwc=(False,False,True)), ifm_prevs = prevs)
    conv_name2 = name + '_conv2'
    network.add(conv_name2,ConvLayer(nifm=nofm //2, nofm=nofm, sofm=sofm, sfil=3))
    return conv_name2

NN = Network('UNet')

NN.set_input_layer(InputLayer(3,224, 224))
inc_out = add_douleConv(NN, 'inc',3,64,224)
down1_out = add_down(NN,'down1', 64,128,112,inc_out)
down2_out = add_down(NN,'down2',128,256,56,down1_out)
down3_out = add_down(NN,'donw3', 256, 512,28, down2_out)
down4_out = add_down(NN,'down4',512,512,14,down3_out)
up1_out = add_up(NN,'up1',1024,512, 28,prevs=(down4_out,down3_out))
up2_out = add_up(NN, 'up2',512,256,56, prevs=(up1_out, down2_out))
up3_out = add_up(NN, 'up3',256,128,112, prevs=(up2_out, down1_out))
up4_out = add_up(NN, 'up4',128,64,224, prevs=(up3_out, inc_out))
NN.add('out_conv', ConvLayer(64,2,224,1), ifm_prevs=up4_out)