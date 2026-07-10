from collections import OrderedDict

from .layer import *

class Network():
    '''
    NN topology. Support DAG structure of layers.
    '''

    def __init__(self, net_name):
        self.INPUT_LAYER_KEY = '__INPUT__'
        self.net_name = net_name
        self.layer_dict = OrderedDict()  ## must use ordered Dict
        self.ifm_prevs_dict = {}     ## record the ifm prevs (layers name), layer dependency in network
        self.wgt_prevs_dict = {}    ## record weight prevs (especially for GEMM)
        self.ext_dict = OrderedDict()  ##

    def set_input_layer(self, input_layer):
        '''
        Set the input layer.
        '''
        if self.INPUT_LAYER_KEY in self.layer_dict:
            raise KeyError('Network: only one input layer is allowed.')

        if not isinstance(input_layer, InputLayer):
            raise TypeError('Network: input_layer must be an InputLayer '
                            'instance.')

        self.layer_dict[self.INPUT_LAYER_KEY] = input_layer

    def input_layer(self):
        '''
        Get the input layer.
        '''
        return self.layer_dict[self.INPUT_LAYER_KEY]

    def add(self, layer_name, layer, ifm_prevs= None, wgt_prevs = None):
        '''
        add a layer to the network, with the its previous layers name
        :param layer_name: strings of adding layer name
        :param layer:
        :param prevs: names of previous layers
        :return:
        '''
        if self.INPUT_LAYER_KEY not in self.layer_dict:
            raise KeyError('Network: only one input layer is allowed.')

        if layer_name in self.layer_dict:
            raise KeyError('Network: layer {} already exists.'
                           .format(layer_name))
        if not isinstance(layer, Layer):
            raise TypeError('Network: layer must be a Layer instance.')

        if ifm_prevs is not None:
            # ensure 'prevs' is tuple
            if isinstance(ifm_prevs,str):
                ifm_prevs = (ifm_prevs,)
            else:
                ifm_prevs = tuple(ifm_prevs)
            # Ensure previous layers are already added.
            for p in ifm_prevs:
                try:
                    self.__getitem__(p)
                except KeyError:
                    raise KeyError('Network: given previous layer {} '
                                   'has not been added to the network'.
                                   format(p))
        ## the last layer in dict is the previous layer of the added layer
        else:
            ifm_prevs = (list(self.layer_dict.keys())[-1],)
    
        self.layer_dict[layer_name] = layer
        self.ifm_prevs_dict[layer_name] = ifm_prevs

        if wgt_prevs is not None:
            # ensure 'prevs' is tuple
            if isinstance(wgt_prevs,str):
                wgt_prevs = (wgt_prevs,)
            else:
                wgt_prevs = tuple(wgt_prevs)
                    # Ensure previous layers are already added.
            for p in wgt_prevs:
                try:
                    self.__getitem__(p)
                except KeyError:
                    raise KeyError('Network: given previous layer {} '
                                   'has not been added to the network'.
                                   format(p))
        self.wgt_prevs_dict[layer_name] = wgt_prevs

    def add_ext(self, layer_name, layer):
        '''
        Add a named external layer.
        '''
        if layer_name in self.ext_dict:
            raise KeyError('Network: external layer {} already exists.'
                           .format(layer_name))
        if not isinstance(layer, InputLayer):
            raise TypeError('Network: external layer must be an InputLayer '
                            'instance.')

        self.ext_dict[layer_name] = layer

    def prevs(self, layer_name):
        '''
        Get the previous layers of the given layer name.

        Return a tuple of all the previous layer names. Use `None` to represent
        the input layer in the returned tuple.
        '''
        if layer_name == self.INPUT_LAYER_KEY:
            raise ValueError('Network: cannot get previous layers for '
                             'input layer.')
        if layer_name in self.ext_dict:
            raise ValueError('Network: cannot get previous layers for '
                             'external layers.')

        ifm_prevs = tuple(None if p == self.INPUT_LAYER_KEY else p
                      for p in self.ifm_prevs_dict[layer_name])
        assert ifm_prevs
        if self.wgt_prevs_dict[layer_name] is not None:
            wgt_prevs = tuple(None if p == self.INPUT_LAYER_KEY else p
                        for p in self.wgt_prevs_dict[layer_name])
        else:
            wgt_prevs = None
        return ifm_prevs, wgt_prevs

    def nexts(self, layer_name):
        '''
        Get the next layers of the given layer name, i.e., the layers that need
        the output of this layer.

        Return a tuple of all the next layer names. Use `None` to represent the
        output of the last layer in the returned tuple.
        '''
        try:
            nexts = tuple(self.nexts_dict[layer_name])
        except KeyError:
            nexts = tuple([None])
        assert nexts

        return nexts

    def firsts(self):
        '''
        Get a tuple of the first layers, i.e., those with only the input layer
        or external layers as their previous layers.

        If a layer has other layers besides the input/external layers as its
        previous layers, it does not count as a first layer.
        '''
        input_ext_layers = set([None]).union(self.ext_layers())
        firsts = []
        for layer_name in self:
            prevs = self.prevs(layer_name)
            if input_ext_layers.issuperset(prevs):
                firsts.append(layer_name)
        return tuple(firsts)

    def lasts(self):
        '''
        Get a tuple of the last layers, i.e., those with no next layer.
        '''
        lasts = []
        for layer_name in self:
            nexts = self.nexts(layer_name)
            if nexts == (None,):
                lasts.append(layer_name)
        return tuple(lasts)

    def ext_layers(self):
        '''
        Get a tuple of the external layers.
        '''
        return tuple(self.ext_dict.keys())

    def _check_prevs(self, layer_name):
        '''
        Check the previous layers of the given layer name.
        '''
        layer = self.layer_dict[layer_name]

        prevs = self.ifm_prevs_dict[layer_name]
        assert prevs

        # Compare the ifmap dimensions of this layer, with all the ofmaps of
        # the previous layers.
        sum_nfmaps = 0

        for p in prevs:
            pl = self.__getitem__(p)

            # Ensure fmap sizes match. Allow padding.
            if not layer.is_valid_padding_sifm((pl.hofm, pl.wofm)):
                raise ValueError('Network: {}, a previous layer of {}, '
                                 'has mismatch fmap size: {} vs. {}.'
                                 .format(p, layer_name,
                                         (pl.hofm, pl.wofm),
                                         (layer.hofm, layer.wofm)))

            sum_nfmaps += pl.nofm

        if sum_nfmaps != layer.nifm:
            raise ValueError('Network: {} cannot be the previous layers of {}.'
                             .format(' | '.join(prevs), layer_name))
    
    def get_mtxu_ops(self, batch_size = 1):
        tot_ops = 0
        for layer_name, layer in self.layer_dict.items():
            if isinstance(layer, ConvLayer) or isinstance(layer, GemmLayer):
                ops_tmp = layer.total_ops(batch_size)
                tot_ops += ops_tmp
        return tot_ops
    
    def get_vecu_ops(self, batch_size =1):
        tot_ops = 0
        for layer_name, layer in self.layer_dict.items():
            if isinstance(layer, LocalRegionLayer):
                ops_tmp = layer.total_ops(batch_size)
                tot_ops += ops_tmp
        return tot_ops

    def get_wei_vol(self, word_size = 1):
        tot_wei_vol = 0
        for layer_name, layer in self.layer_dict.items():
            if isinstance(layer, ConvLayer):
                vol_tmp = layer.total_filter_size(word_size)
                tot_wei_vol += vol_tmp
        return tot_wei_vol
    
    def traverese_layer(self, check = False):
        '''
        generate the Breadth-first-Search DAG layer index
        '''
        self.layer_idx_bfs = OrderedDict()
        # ext_dict layers (e.g. LSTM cells' recurrent-state placeholders added via add_ext())
        # are already-available inputs like INPUT_LAYER_KEY, so seed them as finished too --
        # without this, any layer depending on one can never be classified (its prevs-name is
        # never in layer_finished), and the loop below asserts on the first empty round.
        ext_names = list(self.ext_dict.keys())
        layer_finished = [self.INPUT_LAYER_KEY] + ext_names
        layer_name_list = list(self.layer_dict.keys()).copy()
        layer_name_list.remove(self.INPUT_LAYER_KEY)
        i = 0
        while len(layer_name_list) != len(layer_finished) - 1 - len(ext_names):
            classified = []
            finish_tmp = []
            for layer in layer_name_list:
                if layer not in layer_finished:
                    ifm_prevs = self.ifm_prevs_dict[layer]
                    wgt_prevs = self.wgt_prevs_dict[layer]
                    if wgt_prevs is not None:
                        prevs = ifm_prevs + wgt_prevs
                    else:
                        prevs = ifm_prevs
                    for prevs_name in prevs:
                        if prevs_name not in layer_finished:
                            depth = -1
                            break
                        else:
                            depth = i
                    if depth >=0:
                        classified.append(layer)
                        finish_tmp.append(layer)
            assert len(classified) > 0
            self.layer_idx_bfs[i] = classified
            if check:
                print(f'Depth {i}: {classified}')
            layer_finished += finish_tmp
            i += 1
            self.depth = i


    def __contains__(self, layer_name):
        ''' Whether the network contains a layer. '''
        return layer_name in self.layer_dict or layer_name in self.ext_dict

    def __len__(self):
        ''' Number of layers in the network. '''
        if self.INPUT_LAYER_KEY not in self.layer_dict:
            assert not self.layer_dict
            return 0
        return len(self.layer_dict) - 1

    def __iter__(self):
        ''' Iterate through layer names. '''
        for layer_name in self.layer_dict.keys():
            if layer_name == self.INPUT_LAYER_KEY:
                continue
            yield layer_name

    def __getitem__(self, layer_name):
        ''' Get the layer by name. '''
        try:
            return self.layer_dict[layer_name]
        except KeyError:
            try:
                return self.ext_dict[layer_name]
            except KeyError as e:
                raise KeyError('Network: {} layer not found.'.format(str(e)))

    def __str__(self):
        str_ = 'Network: {}\n'.format(self.net_name)
        for layer_name in self:
            prevs = self.prevs(layer_name)
            prev_str = ' | '.join(['None' if n is None else n for n in prevs])
            str_ += '  Layer {} <- {}\n'.format(layer_name, prev_str)
        return str_

