from core.layer import *
from core.network import Network


def add_attention(network:Network, name:str, len:int, numG:int, gSize:int, prevQ, prevK, prevV):
    prevs = tuple()
    for i in range(numG):
        K_name = name + '_k' + str(i)
        network.add(K_name, ConvLayer(nifm=numG*gSize,nofm=gSize,sofm=(len,1),sfil=1),ifm_prevs=prevK)
        Q_name = name + "_Q" + str(i)
        network.add(Q_name,ConvLayer(nifm=numG*gSize,nofm=gSize,sofm=(len,1),sfil=1),ifm_prevs=prevQ)
        V_name = name + "_V" + str(i)
        network.add(V_name,ConvLayer(nifm=numG*gSize,nofm=gSize,sofm=(len,1),sfil=1),ifm_prevs=prevV)
        QK_name = name + '_QK' + str(i)
        network.add(QK_name, GemmLayer(nifm=gSize, nofm=len, sofm=(len, 1), mtxBT=True),  ifm_prevs=Q_name, wgt_prevs= K_name)
        QKV_name = name + '_QKV' + str(i)
        network.add(QKV_name, GemmLayer(nifm=len, nofm=gSize, sofm=(len, 1), mtxBT=False), ifm_prevs=QK_name, wgt_prevs=V_name)
        prevs += (QKV_name,)
    att_name = name + '_FC'
    network.add(att_name, groupConvLayer(nifm=gSize, nofm=numG*gSize, sofm=(len, 1), sfil=1, numG=numG, concat_hwc=(False, False, True)), ifm_prevs=prevs)
    return att_name

def add_encoder(network:Network, name:str, len:int, numG:int, gSize:int, ff_len:int, prev):
    next_prev = add_attention(network, name, len, numG, gSize, prev, prev, prev)
    etl1_name = name + '_elt1'
    network.add(etl1_name, EltwiseLayer(nofm=numG*gSize, sofm=(len,1), nreg=2), ifm_prevs=(prev, next_prev))
    ff1_name = name + '_feedfwd1'
    network.add(ff1_name, ConvLayer(nifm=numG*gSize, nofm=ff_len,sofm=(len,1),sfil=1), ifm_prevs=etl1_name)
    ff2_name = name + '_feedfwd2'
    network.add(ff2_name, ConvLayer(nifm=ff_len, nofm=numG*gSize, sofm=(len,1), sfil=1),ifm_prevs=ff1_name)
    etl2_name = name + '_elt2'
    network.add(etl2_name, EltwiseLayer(nofm=numG*gSize, sofm=(len, 1), nreg=2), ifm_prevs=(etl1_name,ff2_name))
    return etl2_name

def add_decoder(network:Network, name:str, len:int, numG:int, gSize:int, ff_len:int, prev, enc_prev):
    next_prev = add_attention(network, name+'_1', len, numG, gSize, prev, prev, prev)
    etl1_name = name + '_elt1'
    network.add(etl1_name, EltwiseLayer(nofm = numG*gSize, sofm=(len,1),nreg = 2), ifm_prevs=(prev,next_prev))
    next_prev = add_attention(network, name+'_2', len, numG, gSize, prev, enc_prev, enc_prev)
    etl2_name = name + '_elt2'
    network.add(etl2_name, EltwiseLayer(nofm=numG*gSize, sofm=(len,1), nreg=2),ifm_prevs=(etl1_name,next_prev))
    ff1_name = name + '_feedfwd1'
    network.add(ff1_name, ConvLayer(nifm=numG*gSize, nofm=ff_len, sofm=(len,1), sfil=1),ifm_prevs=etl2_name)
    ff2_name = name + '_feedfwd2'
    network.add(ff2_name, ConvLayer(nifm= ff_len, nofm=numG*gSize, sofm=(len,1),sfil=1), ifm_prevs=ff1_name)
    elt3_name = name + '_elt3'
    network.add(elt3_name, EltwiseLayer(nofm=numG*gSize, sofm=(len,1), nreg=2), ifm_prevs=(etl2_name, ff2_name))
    return elt3_name

    
NN = Network('transformer')

numG = 8
gSize = 64
seq_len = 512
ff_len = 2048
vocab_len = 1000
nEncoder = 4
nDecoder = 4

NN.set_input_layer(InputLayer(numG*gSize, (seq_len, 1)))
enc_prev = 'word_embed_enc'
dec_prev = 'word_embed_dec'
NN.add(enc_prev, PoolingLayer(nofm=numG*gSize, sofm=(seq_len,1), sreg=1), ifm_prevs=NN.INPUT_LAYER_KEY)
NN.add(dec_prev, PoolingLayer(nofm=numG*gSize, sofm=(seq_len,1), sreg=1), ifm_prevs=NN.INPUT_LAYER_KEY)

for i in range(nEncoder):
    enc_prev = add_encoder(NN, 'encoder' + str(i), seq_len, numG, gSize, ff_len, enc_prev)

for i in range(nDecoder):
    dec_prev = add_decoder(NN, 'decoder' + str(i), seq_len, numG, gSize, ff_len, dec_prev, enc_prev)

NN.add('proj', ConvLayer(nifm=numG*gSize, nofm=vocab_len, sofm=(seq_len, 1), sfil=1), dec_prev)



