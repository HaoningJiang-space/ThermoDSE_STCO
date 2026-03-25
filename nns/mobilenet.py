from core.layer import *
from core.network import Network


def InvertedResidual(network:Network, name:str,sifm, nifm, nofm, strd, expand_ratio, prevs=None):
    hidden_dim = nifm * expand_ratio
    identity = strd == 1 and nifm==nofm
    sofm = sifm // strd
    # print(f'{name}, sifm:{sifm}, sofm:{sofm}')
    if expand_ratio == 1:
        lname1 = name + "_dw1"
        network.add(lname1, DPConvLayer(hidden_dim,sofm,3,strd), ifm_prevs= prevs)
        lname_o = name + '_conv2'
        network.add(lname_o, ConvLayer(hidden_dim, nofm,sofm, 1,1 ), ifm_prevs=lname1)
    else:
        lname1 = name + '_conv1'
        network.add(lname1, ConvLayer(nifm, hidden_dim, sifm, 1,1 ), ifm_prevs=prevs)
        lname2 = name + '_dw2'
        network.add(lname2, DPConvLayer(hidden_dim,sofm,3,strd), ifm_prevs= lname1)
        lname_o = name + '_conv2'
        network.add(lname_o, ConvLayer(hidden_dim, nofm,sofm, 1,1 ), ifm_prevs=lname2)
    if identity == False:
        return lname_o
    else:
        network.add(name+'_res', EltwiseLayer(nofm,sofm,2),ifm_prevs=(prevs, lname_o))
        return name+'_res'




NN = Network('mobilenetV2')
cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

NN.set_input_layer(InputLayer(3, 224))

NN.add('conv1', ConvLayer(3, 32, 112, 3, 2))

sifm = 112
input_channel = 32
name_nxt = 'conv1'
for i, [t, c, n, s] in enumerate(cfgs):
    output_channel = c
    blk_name = 'blk' + str(i)
    for j in range(n):
        name = blk_name + '_' + str(j)
        # print(sifm)
        stride = s if j == 0 else 1
        name_nxt = InvertedResidual(NN, name,sifm,input_channel,output_channel,stride, t, prevs=name_nxt)
        input_channel = output_channel
        if stride == 2:
            sifm = sifm // 2

# assert sifm == 7
# NN.add('conv1x1', ConvLayer(320, sifm, 1280, 1, 1))
# NN.add('pooling',PoolingLayer(1280,1,7))

# NN.add('fc', FCLayer(1280, 1000))







