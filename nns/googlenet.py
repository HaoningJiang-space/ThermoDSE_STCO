""" $lic$
Copyright (C) 2016-2020 by Tsinghua University and The Board of Trustees of
Stanford University

This program is free software: you can redistribute it and/or modify it under
the terms of the Modified BSD-3 License as published by the Open Source
Initiative.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the BSD-3 License for more details.

You should have received a copy of the Modified BSD-3 License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

from core.layer import *
from core.network import Network
'''
GoogLeNet

ILSVRC 2014
'''

NN = Network('GoogLeNet')

NN.set_input_layer(InputLayer(3, 224))

NN.add('conv1', ConvLayer(3, 64, 112, 7, 2))
NN.add('pool1', PoolingLayer(64, 56, 3, strd=2))
# Norm layer is ignored.

NN.add('conv2_3x3_reduce', ConvLayer(64, 64, 56, 1))
NN.add('conv2_3x3', ConvLayer(64, 192, 56, 3))
# Norm layer is ignored.
NN.add('pool2', PoolingLayer(192, 28, 3, strd=2))


def add_inception(network, incp_id, sfmap, nfmaps_in, nfmaps_1, nfmaps_3r,
                  nfmaps_3, nfmaps_5r, nfmaps_5, nfmaps_pool, ifm_prevs):
    ''' Add an inception module to the network. '''
    pfx = 'inception_{}_'.format(incp_id)
    # 1x1 branch.
    network.add(pfx + '1x1', ConvLayer(nfmaps_in, nfmaps_1, sfmap, 1),
                ifm_prevs=ifm_prevs)
    # 3x3 branch.
    network.add(pfx + '3x3_reduce', ConvLayer(nfmaps_in, nfmaps_3r, sfmap, 1),
                ifm_prevs=ifm_prevs)
    network.add(pfx + '3x3', ConvLayer(nfmaps_3r, nfmaps_3, sfmap, 3))
    # 5x5 branch.
    network.add(pfx + '5x5_reduce', ConvLayer(nfmaps_in, nfmaps_5r, sfmap, 1),
                ifm_prevs=ifm_prevs)
    network.add(pfx + '5x5', ConvLayer(nfmaps_5r, nfmaps_5, sfmap, 5))
    # Pooling branch.
    network.add(pfx + 'pool_proj', ConvLayer(nfmaps_in, nfmaps_pool, sfmap, 1),
                ifm_prevs=ifm_prevs)
    # Merge branches.
    return (pfx + '1x1', pfx + '3x3', pfx + '5x5', pfx + 'pool_proj')


_ifm_prevs = ('pool2',)

# Inception 3.
_ifm_prevs = add_inception(NN, '3a', 28, 192, 64, 96, 128, 16, 32, 32,
                       ifm_prevs=_ifm_prevs)
_ifm_prevs = add_inception(NN, '3b', 28, 256, 128, 128, 192, 32, 96, 64,
                       ifm_prevs=_ifm_prevs)

NN.add('pool3', PoolingLayer(480, 14, 3, strd=2), ifm_prevs=_ifm_prevs)
_ifm_prevs = ('pool3',)

# Inception 4.
_ifm_prevs = add_inception(NN, '4a', 14, 480, 192, 96, 208, 16, 48, 64,
                       ifm_prevs=_ifm_prevs)
_ifm_prevs = add_inception(NN, '4b', 14, 512, 160, 112, 224, 24, 64, 64,
                       ifm_prevs=_ifm_prevs)
_ifm_prevs = add_inception(NN, '4c', 14, 512, 128, 128, 256, 24, 64, 64,
                       ifm_prevs=_ifm_prevs)
_ifm_prevs = add_inception(NN, '4d', 14, 512, 112, 144, 288, 32, 64, 64,
                       ifm_prevs=_ifm_prevs)
_ifm_prevs = add_inception(NN, '4e', 14, 528, 256, 160, 320, 32, 128, 128,
                       ifm_prevs=_ifm_prevs)

NN.add('pool4', PoolingLayer(832, 7, 3, strd=2), ifm_prevs=_ifm_prevs)
_ifm_prevs = ('pool4',)

# Inception 5.
_ifm_prevs = add_inception(NN, '5a', 7, 832, 256, 160, 320, 32, 128, 128,
                       ifm_prevs=_ifm_prevs)
_ifm_prevs = add_inception(NN, '5b', 7, 832, 384, 192, 384, 48, 128, 128,
                       ifm_prevs=_ifm_prevs)

NN.add('pool5', PoolingLayer(1024, 1, 7), ifm_prevs=_ifm_prevs)

NN.add('fc', FCLayer(1024, 1000))

