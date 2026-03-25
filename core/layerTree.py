import numpy as np


class Layer():
    '''
    Base layer class
        ch: channel number of feature map
        sofm: shape of ouput feature map list/INT, 0: H, 1: W
    '''

    def __init__(self, nofm, sofm, strd=1):
        if isinstance(sofm, int):
            hofm = sofm
            wofm = sofm
        elif len(sofm) == 2:
            hofm = sofm[0]
            wofm = sofm[1]
        else:
            raise ValueError('Layer: sofm is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sofm))
        assert hofm > 0 and wofm > 0

        if isinstance(strd, int):
            htrd = strd
            wtrd = strd
        elif len(strd) == 2:
            htrd = strd[0]
            wtrd = strd[1]
        else:
            raise ValueError('Layer: strd is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(strd))
        assert htrd > 0 and wtrd > 0

        self.nofm = nofm
        self.hofm = hofm
        self.wofm = wofm

        self.htrd = htrd
        self.wtrd = wtrd

    def input_layer(self):
        ''' Get the input layer parameters. The ifm info is from this '''
        raise NotImplementedError(self.__class__.__name__)

    def nifm(self):
        return self.input_layer().nofm  # channel of IFM

    def hifm(self):
        return self.input_layer().hifm  # Height of IFM

    def wifm(self):
        return self.input_layer().wifm  # Weight of IFM

    def ofmap_size(self, batch_size=1, word_size=1):
        '''
        get the ofmap size of pre output channel for partitions
        '''
        return self.hofm * self.wofm * batch_size * word_size

    def total_ofmap_size(self, batch_size=1, word_size=1):
        '''
        get the totally ofmap size
        '''
        return self.nofm * self.ofmap_size(batch_size, word_size)

    def ifm_size(self, batch_size=1, word_size=1):
        '''
        Get size of  input fmap of one channel with `batch_size`.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.input_layer().ofmap_size(batch_size, word_size)

    def total_ifmap_size(self, batch_size=1, word_size=1):
        '''
        Get size of  input fmap of one channel with `batch_size`.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.input_layer().total_ofmap_size(batch_size, word_size)

    def ops_per_neuron(self):
        ''' Number of operations per neuron. '''
        raise NotImplementedError(self.__class__.__name__)

    def total_ops(self, batch_size=1):
        ''' Get total number of operations. '''
        return self.total_ofmap_size() * self.ops_per_neuron() * batch_size

    def is_valid_padding_sifm(self, sifm):
        ''' Whether the given `sifm` is valid when allowing padding. '''
        if isinstance(sifm, int):
            hifm = sifm
            wifm = sifm
        elif len(sifm) == 2:
            hifm = sifm[0]
            wifm = sifm[1]
        else:
            raise ValueError('Layer: sifm is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sifm))

        h_padding_rng = sorted((self.hofm * self.htrd, self.hifm))
        w_padding_rng = sorted((self.wofm * self.wtrd, self.wifm))
        return (h_padding_rng[0] <= hifm <= h_padding_rng[1]
                and w_padding_rng[0] <= wifm <= w_padding_rng[1])

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'strd={}'.format(repr((self.htrd, self.wtrd)))]))


class InputLayer(Layer):
    '''
    NN input layer parameters.
    '''

    @staticmethod
    def input_layer(self):
        return None

    def ops_per_neuron(self):
        return 0


class ConvLayer(Layer):
    '''
    NN convolutional layer parameters.

    nifm : # ifmap channels C
    nofm : # ofmap channels K
    hifm, wifm : ifmap height/width
    hofm, wofm : ofmap height/width
    hfil, wfil : weight filter width/height
    htrd, wtrd : stride height/width
    '''

    def __init__(self, nifm, nofm, sofm, sfil, strd=1):
        super(ConvLayer, self).__init__(nofm, sofm, strd=strd)
        if isinstance(sfil, int):
            hfil = sfil
            wfil = sfil
        elif len(sfil) == 2:
            hfil = sfil[0]
            wfil = sfil[1]
        else:
            raise ValueError('ConvLayer: sfil is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sfil))

        self.hfil = hfil
        self.wfil = wfil
        # this setting includes the zero mapping of IFM,
        # hifm = self.hfil + (self.hofm - 1) * self.htrd
        # wifm = self.wfil + (self.wofm - 1) * self.wtrd
        self.hifm = self.hofm * self.htrd + (hfil  - 1)
        self.wifm = self.wofm * self.wtrd + (wfil - 1)
        self.inlayer = Layer(nifm, (self.hifm, self.wifm))
        self.nifm = nifm

    def input_layer(self):
        return self.inlayer

    def ops_per_neuron(self):
        # 2D convolution across all ifmap channels.
        return self.hfil * self.wfil * self.nifm

    def filter_size(self, word_size=1):
        '''
        Get size of one weight filter.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.hfil * self.wfil * word_size

    def total_filter_size(self, word_size=1):
        '''
        Get total size of all weight filters.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.nifm * self.nofm * self.filter_size(word_size)

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nifm={}'.format(repr(self.nifm)),
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'sfil={}'.format(repr((self.hfil, self.wfil))),
                'strd={}'.format(repr((self.htrd, self.wtrd)))]))


class FCLayer(ConvLayer):
    '''
    NN fully-connected layer parameters.

    As a special case of CONVLayer.

    hifm = hfil, wifm = wfil, strd = 1, hofm = wofm = 1
    '''

    def __init__(self, nifm, nofm, sfil=1):
        super(FCLayer, self).__init__(nifm, nofm, 1, sfil)
        assert self.hofm == 1 and self.wofm == 1

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nifm={}'.format(repr(self.nifm)),
                'nofm={}'.format(repr(self.nofm)),
                'sfil={}'.format(repr((self.hfil, self.wfil)))]))


class LocalRegionLayer(Layer):
    '''
    NN layer which computes on a local region. The layer has no or limited
    shared weights
    nofm: # ofmap channels K
    sofm: shape of fmap  H and W
    nreg: # channel of kernel/filter (in channel dimension) (e.g. pooling layer is nreg = 1)
    sreg: shape of kernel/filter (in channel dimension)
    ntrd: the kernel/fileter stride along channel dimension
    strd: stride along H and W dimension
    Includes pooling layer, normalization layer, and element-wise layer, post-process layer in attention layer(to be done).
    '''

    def __init__(self, nofm, sofm, nreg, sreg, ntrd=1, strd=1):
        super(LocalRegionLayer, self).__init__(nofm, sofm, strd=strd)

        if isinstance(sreg, int):
            hreg = sreg
            wreg = sreg
        elif len(sreg) == 2:
            hreg = sreg[0]
            wreg = sreg[1]
        else:
            raise ValueError('LocalRegionLayer: sreg is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sreg))
        if nreg > 1 and (hreg * wreg) > 1:
            raise ValueError('LocalRegionLayer: local region cannot be a mix '
                             'of both n ({}) and h & w ({}, {})'
                             .format(nreg, hreg, wreg))
        self.nreg = nreg
        self.hreg = hreg
        self.wreg = wreg
        self.ntrd = ntrd

        nifm = self.nofm * self.ntrd  # ignore all-zero padding channels.
        # this setting includes the zero mapping of IFM,
        # but the zero padding data will not move between different cores and not store in on buffers
        # hifm = self.hreg + (self.hofm - 1) * self.htrd
        # wifm = self.wreg + (self.wofm - 1) * self.wtrd
        hifm = self.hofm  * self.htrd
        wifm = self.wofm  * self.wtrd
        self.inlayer = Layer(nifm, (hifm, wifm))
        self.hifm = hifm
        self.wifm = wifm
        self.nifm = nifm

    def input_layer(self):
        return self.inlayer

    def ops_per_neuron(self):
        # Each output point corresponds to merging a local region.
        return self.region_size()

    def region_size(self):
        ''' The size of the local region corresponding to one output point. '''
        return self.nreg * self.hreg * self.wreg

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'nreg={}'.format(repr(self.nreg)),
                'sreg={}'.format(repr((self.hreg, self.wreg))),
                'ntrd={}'.format(repr(self.ntrd)),
                'strd={}'.format(repr((self.htrd, self.wtrd)))]))


class PoolingLayer(LocalRegionLayer):
    '''
    NN pooling layer parameters.

    As a special case of LocalRegionLayer.

    nreg = ntrd = 1
    '''

    def __init__(self, nofm, sofm, sreg, strd=None):
        if strd is None:
            strd = sreg
        super(PoolingLayer, self).__init__(nofm, sofm, 1, sreg,
                                           ntrd=1, strd=strd)
        assert self.nreg == 1
        assert self.ntrd == 1

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nifm={}'.format(repr(self.nifm)),
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'sreg={}'.format(repr((self.hreg, self.wreg))),
                'strd={}'.format(repr((self.htrd, self.wtrd)))]))


class EltwiseLayer(LocalRegionLayer):
    '''
    NN element-wise layer parameters.

    As a special case of LocalRegionLayer.

    nreg = ntrd, sreg = 1
    '''

    def __init__(self, nofm, sofm, nreg):
        super(EltwiseLayer, self).__init__(nofm, sofm, nreg, 1,
                                           ntrd=nreg, strd=1)
        assert self.hreg == self.wreg == 1

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nifm={}'.format(repr(self.nifm)),
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'nreg={}'.format(repr(self.nreg))]))
