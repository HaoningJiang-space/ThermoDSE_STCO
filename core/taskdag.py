from collections import OrderedDict

from .util import Range, FmapRange, get_part_range, get_part_range_limit, ofm_to_ifm
from core import Network, ConvLayer, LocalRegionLayer, GemmLayer, groupConvLayer, DPConvLayer
from core import Layer
from copy import deepcopy


# class Edge():
#     def __init__(self, ):

class TaskNode():
    def __init__(self, layer_name, fromrange: FmapRange, torange: FmapRange, padding_rate = 0):
        self.fromrange = fromrange
        self.torange = torange
        self.ofm_valid_time = 0  # the OFM will not be used after this execute order, (will not occupy ubuf)
        self.ofm_wrb_time = None  # when the OFM write back to DRAM
        self.ofm_exe_time = None
        self.map_coreIdx = None
        self.layer_name = layer_name
        self.padding_rate = padding_rate
        # TO Do: all the sparsity of IFM to get the more accurate energy cost

    # IFM size
    def fromVolume(self, word_size=1):
        return self.fromrange.volume() * word_size

    # OFM size
    def toVolume(self, word_size=1):
        return self.torange.volume() * word_size

    def set_map_coreIdx(self, coreIdx):
        self.map_coreIdx = coreIdx

    def get_map_coreIdx(self):
        return self.map_coreIdx

    def set_valid_time(self, order):
        if self.ofm_exe_time is None or order < self.ofm_exe_time:
            raise ValueError('OFM will not be generated before order {}.'.format(order))
        self.ofm_valid_time = order

    def get_valid_time(self):
        return self.ofm_valid_time

    def set_exe_time(self, exe_order):
        self.ofm_exe_time = exe_order

    def get_exe_time(self):
        return self.ofm_exe_time

    def set_writeback_time(self, exe_order):
        self.ofm_wrb_time = exe_order

    def get_writeback_time(self):
        return self.ofm_wrb_time

    def is_ofm_dram(self, order_cur):
        if order_cur > self.ofm_valid_time:
            raise ValueError('OFM of task in {} is not valid, why query this?'.format(self.layer_name))
        if self.ofm_exe_time is None:
            raise ValueError('OFM is not generated at order {}.'.format(order_cur))
        if self.ofm_wrb_time is None:
            return True
        else:
            return order_cur < self.ofm_wrb_time

    def total_filter_size(self, word_size=1):
        return 0

    def ops_per_neuron(self):
        ''' Number of operations per neuron. '''
        raise NotImplementedError(self.__class__.__name__)

    def total_ops(self):
        ''' Get total number of operations. '''
        return self.toVolume() * self.ops_per_neuron() 

    def workload_cmr(self, word_size = 1):
        return self.total_ops() / (self.fromVolume(word_size) + self.total_filter_size(word_size))

    def max_reuse_volume(self, othernode):
        return max(self.ifm_reuse_volume(othernode), self.wei_reuse_volume(othernode))

    def ifm_reuse_volume(self, othernode):
        return 0

    def wei_reuse_volume(self, othernode):
        return 0

    def get_padding_rate(self):
        return self.padding_rate

    def get_ofm_shape(self):
        return self.torange.bRange.length(), self.torange.hRange.length(), self.torange.wRange.length(), self.torange.cRange.length()

    def __str__(self):
        str_ = ('In layer {}, From range {}, To range {}, ofm_valid_time: {}, ofm_exe_time: {}, ofm_wrb_time: {} map '
                'core: {} ').format(self.layer_name, str(self.fromrange), str(self.torange),
                                    str(self.ofm_valid_time), str(self.ofm_exe_time),
                                    str(self.ofm_wrb_time), str(self.map_coreIdx))
        return str_

class GemmTaskNode(TaskNode):
    def __init__(self, layer_name, layer, mtxArange: FmapRange, mtxBrange: FmapRange, torange: FmapRange, padding_rate = 0):
        super(GemmTaskNode, self).__init__(layer_name, mtxArange, torange, padding_rate)
        self.hfil = 1
        self.wfil = 1
        self.mtxBT = layer.mtxBT
        self.mtxBrange = mtxBrange
        if self.mtxBT == False and mtxArange.cRange.length() != mtxBrange.hRange.length() * mtxBrange.wRange.length():
            raise ValueError(f'When mtxB not transpose, C range in mtxA is {mtxArange.cRange.length()}, which is different with Hrange * Wrange in mtxB {mtxBrange.hRange.length() * mtxBrange.wRange.length()}')
    
    def total_filter_size(self, word_size = 1):
        return self.mtxBrange.volume() * word_size
    
    def mtxA_reuse_volume(self, othernode):
        '''
        determine if ifm in this node can be reused by other node
        '''
        if self.layer_name != othernode.layer_name:
            return 0

        if self.fromrange.bRange.isSame(othernode.fromrange.bRange) and self.fromrange.hRange.isSame(othernode.fromrange.hRange) and self.fromrange.wRange.isSame(
                othernode.fromrange.wRange) and (not self.torange.cRange.isSame(othernode.torange.cRange)):
            return self.fromVolume()
        else:
            return 0
        
    def ops_per_neuron(self):
        return self.mtxBrange.cRange.length()
    
    def get_mtxB_range(self):
        if self.mtxBT:
            brange = self.mtxBrange.bRange
            hrange = self.mtxBrange.cRange
            crange = self.mtxBrange.hRange
            return FmapRange(brange, hrange, self.mtxBrange.wRange, crange)
        else:
            return self.mtxBrange

        
    def mtxB_reuse_volume(self,othernode):
        if self.layer_name != othernode.layer_name:
            return 0
        if (not self.fromrange.hRange.isSame(othernode.fromrange.hRange)) and (
                not self.fromrange.wRange.isSame(othernode.fromrange.wRange)) and self.torange.cRange.isSame(
            othernode.torange.cRange) and self.fromrange.bRange.isSame(self.fromrange.bRange):
            return self.total_mtxB_size()
        else:
            return 0
    
    def ifm_hw_overlap(self):
        return 1


class CONVTaskNode(TaskNode):
    def __init__(self, layer_name, layer, fromrange: FmapRange, torange: FmapRange, padding_rate = 0):
        super(CONVTaskNode, self).__init__(layer_name, fromrange, torange, padding_rate)
        self.hfil = layer.hfil
        self.wfil = layer.wfil
        self.htrd = layer.htrd
        self.wtrd = layer.wtrd

    def filter_size(self, word_size=1):
        return self.hfil * self.wfil * word_size

    def total_filter_size(self, word_size=1):
        return self.ops_per_neuron() * self.torange.cRange.length() * word_size

    def ops_per_neuron(self):
        return self.hfil * self.wfil * self.fromrange.cRange.length() 

    def ifm_reuse_volume(self, othernode):
        if self.layer_name != othernode.layer_name:
            return 0

        if self.fromrange.bRange.isSame(othernode.fromrange.bRange) and self.fromrange.hRange.isSame(othernode.fromrange.hRange) and self.fromrange.wRange.isSame(
                othernode.fromrange.wRange) and (not self.torange.cRange.isSame(othernode.torange.cRange)):
            return self.fromVolume()
        else:
            return 0

    def wei_reuse_volume(self, othernode):
        if self.layer_name != othernode.layer_name:
            return 0
        if (not self.fromrange.hRange.isSame(othernode.fromrange.hRange)) and (
                not self.fromrange.wRange.isSame(othernode.fromrange.wRange)) and self.torange.cRange.isSame(
            othernode.torange.cRange):
            return self.total_filter_size()
        else:
            return 0

    def ifm_hw_overlap(self):
        return self.hfil * self.wfil / self.htrd / self.wtrd

class DPConvTaskNode(CONVTaskNode):
    def __init__(self, layer_name, layer, fromrange: FmapRange, torange: FmapRange, padding_rate = 0):
        super(DPConvTaskNode, self).__init__(layer_name, layer,fromrange, torange, padding_rate)

    def ops_per_neuron(self):
        return self.hfil * self.wfil * 1 

class LRLTaskNode(TaskNode):
    def __init__(self, layer_name, layer, fromrange: FmapRange, torange: FmapRange, padding_rate = 0):
        super(LRLTaskNode, self).__init__(layer_name, fromrange, torange, padding_rate)
        self.nreg = layer.nreg
        self.hreg = layer.hreg
        self.wreg = layer.wreg

    def fromVolume(self, word_size=1):
        return self.fromrange.volume() * word_size * self.nreg

    def region_size(self):
        return self.nreg * self.hreg * self.wreg

    def filter_size(self, word_size=1):
        return 0

    def total_filter_size(self, word_size=1):
        return 0

    def ops_per_neuron(self):
        return self.region_size()

    def wei_reuse_volume(self, othernode):
        return 0

    def ifm_reuse_volume(self, othernode):
        return 0


class TaskDAG():
    def __init__(self, net_name):
        self.net_name = net_name
        self.task_dict = OrderedDict()
        self.prevs_dict = {}  # for schedule
        self.child_dict = {}  # for mapping child[parent][child] = overlapVolume

    def add_task(self, taskNode, name, prevs=None):
        '''
        add a taskNode into the taskDAG, with previous task name
        :param taskNode: task node
        :param prevs: the previous tasks name
        :return:
        '''
        if name in self.task_dict:
            raise KeyError(f'taskNode {name} already exists')

        if prevs is not None:
            # ensure 'prevs' is tuple
            if isinstance(prevs, str):
                prevs = (prevs,)
            else:
                prevs = tuple(prevs)
            # Ensure previous layers are already added.
            for p in prevs:
                try:
                    self.__getitem__(p)
                except KeyError:
                    raise KeyError('TaskNode: given previous node {} '
                                   'has not been added to the taskdag'.
                                   format(p))
        self.task_dict[name] = taskNode

    def add_prevs(self, node_name, prevs):
        if node_name not in self.task_dict:
            raise KeyError(f'task node {node_name} does not exist')

        if prevs is not None:
            # ensure 'prevs' is tuple
            if isinstance(prevs, str):
                prevs = (prevs,)
            else:
                prevs = tuple(prevs)
            # make sure all prevs have been added
            for p in prevs:
                if p not in self.prevs_dict:
                    raise KeyError(f'previous node {p} does not exist in task DAG')
            # print(prevs)
        if len(prevs) == 0:
            prevs = ('__INPUT_LAYER__',)
        self.prevs_dict[node_name] = prevs

    def add_child(self, parent_name, child_name, overlap_fmap):
        if parent_name not in self.child_dict:
            raise KeyError(f'parent node {parent_name} does not exist')

        if child_name in self.child_dict[parent_name]:
            raise KeyError(f'For {parent_name} child node: {child_name} already exists')

        self.child_dict[parent_name][child_name] = overlap_fmap

    def create_nodes(self, nn: Network, part_sch_list, batch):
        '''
        create task according to the nn and part_sch_list
        :param nn: network
        :param part_sch_list: part_sch_list from part engine
        :return:
        '''
        for i, lname in enumerate(nn):
            layer = nn[lname]
            part_sch = part_sch_list[lname]
            h_range = get_part_range(layer.hofm, part_sch[0])
            w_range = get_part_range(layer.wofm, part_sch[1])
            n_range = get_part_range(layer.nofm, part_sch[2])
            # cle(n_range)
            for bs in range(batch):
                for n_part in range(len(n_range) - 1):
                    for w_part in range(len(w_range) - 1):
                        for h_part in range(len(h_range) - 1):
                            node_name = lname +'.' + str(bs) +'.' + str(h_part) + '.' + str(w_part) + '.' + str(n_part)
                            bofm_range = Range(bs, bs+1)
                            hofm_range = Range(h_range[h_part], h_range[h_part + 1])
                            wofm_range = Range(w_range[w_part], w_range[w_part + 1])
                            nofm_range = Range(n_range[n_part], n_range[n_part + 1])
                            to_range = FmapRange(bofm_range, hofm_range, wofm_range, nofm_range)
                            hifm_range, wifm_range, padding_rate = ofm_to_ifm(layer, hofm_range, wofm_range)
                            # if node_name == 'conv5_0_b.0.0.1.0':
                            #     print(f'{lname} : OFM range:{to_range}, hifm:{hifm_range}, wifm:{wifm_range}')
                            if isinstance(layer, ConvLayer) and not isinstance(layer, groupConvLayer) and not isinstance(layer, DPConvLayer) :
                                nifm_range = Range(0, layer.get_nifm())
                                from_range = FmapRange(bofm_range, hifm_range, wifm_range, nifm_range)
                                task = CONVTaskNode(lname, layer, from_range, to_range, padding_rate)
                            elif isinstance(layer, DPConvLayer):
                                nifm_range = nofm_range
                                from_range = FmapRange(bofm_range, hifm_range, wifm_range, nifm_range)
                                task = DPConvTaskNode(lname, layer, from_range, to_range, padding_rate)
                            elif isinstance(layer, groupConvLayer):
                                nifm_range = Range(0, layer.get_nifm())
                                from_range = FmapRange(bofm_range, hifm_range, wifm_range, nifm_range)
                                for ii in range(layer.numG):
                                    node_name = lname + '.' + str(bs) + '.' + str(h_part) + '.' + str(w_part) + '.' + str(n_part) + '.' + str(ii)
                                    task = CONVTaskNode(lname, layer, from_range, to_range, padding_rate)
                                    self.add_task(task, node_name)
                                    self.child_dict[node_name] = {}
                            elif isinstance(layer, LocalRegionLayer):
                                nifm_range = Range(n_range[n_part], n_range[n_part + 1]) if layer.up == 0 else Range(n_range[n_part] // layer.ntrd, n_range[n_part + 1]// layer.ntrd)
                                from_range = FmapRange(bofm_range, hifm_range, wifm_range, nifm_range)
                                task = LRLTaskNode(lname, layer, from_range, to_range, padding_rate)
                            elif isinstance(layer, GemmLayer):
                                nifm_range = Range(0, layer.get_nifm())
                                mtxA_range = FmapRange(bofm_range, hifm_range, wifm_range, nifm_range)
                                hwgt_range = Range(0, layer.get_nifm())
                                wwgt_range = Range(0, 1)
                                nifm_range = nofm_range
                                mtxB_range = FmapRange(bofm_range, hwgt_range, wwgt_range, nofm_range)
                                task = GemmTaskNode(lname, layer, mtxA_range, mtxB_range, to_range, padding_rate)
                            else:
                                raise TypeError('layer type not supported')
                            if not isinstance(layer, groupConvLayer):
                                self.add_task(task, node_name)
                                self.child_dict[node_name] = {}
                            # print(task)

    def create_edges(self, nn: Network):
        for node_name in self.task_dict.keys():
            task_node = self.task_dict[node_name]
            if 'blk1_0_dw1.0.3.0.0' in node_name:
                print(1)
            prevs = tuple()
            # print(f'ifm range: {task_node.fromrange}')
            for name_tmp in self.task_dict.keys():  ## traverse all task node to find prevs
                if name_tmp != node_name:
                    task_node_tmp = self.task_dict[name_tmp]
                    layer_name1 = task_node.layer_name
                    layer_name2 = task_node_tmp.layer_name
                    ifm_prevs, wgt_prevs = nn.prevs(layer_name1)
                    # print(ifm_prevs, wgt_prevs)
                    if layer_name2 in ifm_prevs:
                        if isinstance(nn[layer_name1], groupConvLayer):
                            assert len(ifm_prevs) == nn[layer_name1].numG
                            gIdx = int(node_name.split('.')[-1])
                            if layer_name2 in ifm_prevs :
                                prevs += (name_tmp,)
                                if task_node.fromrange.dependent(task_node_tmp.torange):
                                    overlap_fmap = task_node.fromrange.overlap_fmap(task_node_tmp.torange)
                                else:
                                    ## empty range to force the groupconv executed after all layer finished
                                    overlap_fmap = FmapRange(Range(0,1), Range(0,1), Range(0,1), Range(0,1))
                                self.add_child(name_tmp, node_name, overlap_fmap)
                        elif task_node.fromrange.dependent(task_node_tmp.torange):
                            # print(f'prevs ofm range: {task_node_tmp.torange}')
                            prevs += (name_tmp,)
                            overlap_fmap = task_node.fromrange.overlap_fmap(task_node_tmp.torange)
                            self.add_child(name_tmp, node_name, overlap_fmap)
                            # if node_name == 'pool1_0_0_0':
                            #     print(node_name + ': ' + str(task_node.fromrange) +name_tmp + ': ' + str(task_node_tmp.torange))
                    if isinstance(task_node, GemmTaskNode):
                        mtxBrange = task_node.get_mtxB_range()
                        if layer_name2 in wgt_prevs:
                            if mtxBrange.dependent(task_node_tmp.torange):
                                prevs += (name_tmp,)
                                overlap_mtxB = mtxBrange.overlap_fmap(task_node_tmp.torange)
                                self.add_child(name_tmp, node_name, overlap_mtxB)
            self.add_prevs(node_name, prevs)

    def generate_taskdag(self, nn: Network, part_sch_list, batch):
        '''
        generate a task dag according to the nn and part_sch_list
        :param nn: network
        :param part_sch_list: part_sch_list from part engine
        '''
        self.create_nodes(nn, part_sch_list, batch)
        self.create_edges(nn)

    def traverse_taskdag(self):
        '''
        generate the Breadth-First-Search DAG traverse
        :return: deepth of taskDAG
        '''
        ## init
        self.task_dfs = OrderedDict()
        node_finish = ['__INPUT_LAYER__']
        task_name_list = list(self.task_dict.keys()).copy()
        i = 0
        while len(task_name_list) != len(node_finish) - 1:
            classfied = []
            finish_tmp = []
            for node_name in task_name_list:
                if node_name not in node_finish:
                    prevs = self.prevs_dict[node_name]
                    # if i == 0:
                    #     print(node_name, prevs)
                    for prevs_name in prevs:
                        if prevs_name not in node_finish:
                            # print(node_name, prevs_name, node_finish)
                            depth = -1
                            # print(depth)
                            break
                        else:
                            # print(node_name, prevs_name, node_finish)
                            depth = i
                        # print(depth)
                    if depth >= 0:
                        # print(deepth, node_name, prevs)
                        classfied.append(node_name)
                        finish_tmp.append(node_name)
            assert len(classfied) > 0
            # print('depth: {}, classfied: {}'.format(i, classfied))
            self.task_dfs[i] = classfied
            # print(f'Depth {i}: {classfied}')
            node_finish += finish_tmp
            self.depth = i
            i = i + 1
        return self.depth
    
    def check_thermal_char(self):
        for deep in range(self.depth):
            tasks = self.task_dfs[deep]
            data_vol = []
            name_stats = {}
            for task_name in tasks:
                task_node = self.task_dict[task_name]
                layer_name = task_name.split('.')[0]
                if layer_name not in name_stats.keys():
                    name_stats[layer_name] = 0
                else:
                    name_stats[layer_name] += 1
                ofm_volume = task_node.toVolume()
                ifm_volume = task_node.fromVolume()
                wei_volume = task_node.total_filter_size()
                data_vol.append(ofm_volume + ifm_volume + wei_volume)
            print(f'layer{name_stats} is in DAG depth: {deep}, the data volume: max {max(data_vol)}, min {min(data_vol)}, ratio:{max(data_vol)/min(data_vol)}')

    def get_node_prevs(self, node_name):
        return self.prevs_dict[node_name]

    def get_node_children(self, node_name):
        return self.child_dict[node_name]

    def set_task_exe_time(self, node_name, exe_order):
        self.task_dict[node_name].set_exe_time(exe_order)

    def set_task_writeback_time(self, node_name, exe_order):
        self.task_dict[node_name].set_writeback_time(exe_order)

    def set_task_valid_time(self, node_name, order):
        self.task_dict[node_name].set_valid_time(order)

    def set_task_coreIdx(self, node_name, core_idx):
        self.task_dict[node_name].set_map_coreIdx(core_idx)

    def get_task_coreIdx(self,node_name):
        return self.task_dict[node_name].get_map_coreIdx()
    def get_task_exe_time(self, node_name):
        return self.task_dict[node_name].get_exe_time()

    def get_task_writeback_time(self, node_name):
        return self.task_dict[node_name].get_writeback_time()

    def get_task_valid_time(self, node_name):
        return self.task_dict[node_name].get_valid_time()

    def get_ofm_volume(self,node_name, word_size = 1):
        return self.task_dict[node_name].toVolume(word_size)

    def get_ifm_volume(self, node_name, word_size = 1):
        return self.task_dict[node_name].fromVolume(word_size)

    def get_wei_volume(self, node_name, word_size = 1):
        return self.task_dict[node_name].total_filter_size(word_size)

    def is_task_ofm_dram(self, node_name, order_cur):
        self.task_dict[node_name].is_ofm_dram(order_cur)

    def get_inter_layer_reuse(self, pname, cname):
        children = self.child_dict[pname]
        if cname not in children.keys():
            return 0
        else:
            return children[cname].volume()

    def get_ubuf_reuse(self, node_name, stored_list:list, verbose=False):
        reuse_vol = 0
        for pname in stored_list:
            reuse_vol_tmp = self.get_inter_layer_reuse(pname, node_name)
            reuse_vol += reuse_vol_tmp
            if verbose:
                print(f'For {node_name}: the buffering ofm {pname} with volume {reuse_vol_tmp} can be reuse ')
        return reuse_vol

    def get_task_ubuf_occpuy(self, node_name):
        vifm = self.task_dict[node_name].fromVolume()
        vwei = self.task_dict[node_name].total_filter_size()
        vofm = self.task_dict[node_name].toVolume()
        return vifm + vwei + vofm

    def clear(self):
        self.task_dict.clear()
        self.prevs_dict.clear()
        self.child_dict.clear()


    def __contains__(self, node_name):
        return node_name in self.task_dict

    def __len__(self):
        return len(self.task_dict)

    def __getitem__(self, node_name):
        ''' Get the task by name. '''
        try:
            return self.task_dict[node_name]
        except KeyError as e:
            raise KeyError('Network: {} layer not found.'.format(str(e)))

    def __iter__(self):
        for node_name in self.task_dict.keys():
            yield node_name
