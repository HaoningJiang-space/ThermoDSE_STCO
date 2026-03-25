
import math
import numpy as np
from .util import *
class Onchip_buffer():
    def __init__(self, name, size, rcost, wcost, area, bw):
        self.size = size
        self.rcost = rcost
        self.wcost = wcost
        self.area = area
        self.name = name
        self.bw = bw

    def get_rcost(self):
        return self.rcost

    def get_wcost(self):
        return self.wcost

    def get_area(self):
        return self.area

    def get_size(self):
        return self.size

    def get_bw(self):
        return self.bw

    def __str__(self):
        str_ = ('size: ' + str(self.size) + 'KByte, energy r/w cost: ' + str(self.rcost) + '/' + str(self.wcost)
                + 'pJ/Byte' + 'area: ' + str(self.area) + ' um^2' + '\n')
        return str_
class Core_mem_hierarchy():
    '''
    memory heiracy
    '''
    def __init__(self):
        self.memory = {}

    def add_memory(self, ubf:Onchip_buffer):
        self.memory[ubf.name] = ubf

    def get_buf_rcost(self, buf_name):
        return self.memory[buf_name].get_rcost()

    def get_buf_wcost(self, buf_name):
        return self.memory[buf_name].get_wcost()

    def get_buf_area(self, buf_name):
        return self.memory[buf_name].get_area()

    def get_buf_size(self, buf_name):
        return self.memory[buf_name].get_size()

    def get_buf_bw(self, buf_name):
        return self.memory[buf_name].get_bw()



## the architecture of one DNN core, target architecture is TPU, NVDLA-style Accelerator
## hence, the parallel is cin-cout (K-C parallel in some paper)
class AccCore():
    def __init__(self, smtxu, svecu, mtxu_cost, vecu_cost, core_mem: Core_mem_hierarchy):
        '''
        :param smtxu: size of matrix unit (Systolic array nxn, e.g. 32x32, (32x32))
        :param svecu: size of vector unit (Considering bw of l1 buf, usually nx1, e.g. 32x1)
        :param core_mem:  Core_mem_hierarchy in accelerator core
        :return:
        '''
        # dnn accelerator core initial info
        if isinstance(smtxu, int):
            self.h_pe = smtxu
            self.w_pe = smtxu
        elif len(smtxu) == 2:
            self.h_pe = smtxu[0]
            self.w_pe = smtxu[1]
        else:
            raise ValueError('size of matrix unit invalid ({}), '
                             'needs to be an int or '
                             'a pair of integers'.format(smtxu))
        self.svecu = svecu
        self.mtxu_cost = mtxu_cost
        self.vecu_cost = vecu_cost
        self.core_mem = core_mem

    def get_buf_rcost(self, buf_name):
        return self.core_mem.get_buf_rcost(buf_name)

    def get_buf_wcost(self, buf_name):
        return self.core_mem.get_buf_wcost(buf_name)

    def get_buf_area(self, buf_name):
        return self.core_mem.get_buf_area(buf_name)

    def get_buf_size(self, buf_name):
        return self.core_mem.get_buf_size(buf_name)

    def get_buf_bw(self, buf_name):
        return self.core_mem.get_buf_bw(buf_name)

    def cal_mtxu_utl(self,sfmap):
        '''
        :param sfmap: shape of feature map both IFM, OFM. formate: (hofm, wofm, nifm, nofm)
        :return: the utilization in 3 dimensions (M, N, K)
        # 3 dimensions utilization in MNK matrix mutilplication, in each round:
        # M (ho* wo and h_pe) :  (hofm * wofm) / h_pe
        # K (nifm and h_pe) :  nifm/h_pe
        # N (nofm and w_pe) : nofm/w_pe
        '''
        hofm = sfmap[0]
        wofm = sfmap[1]
        nifm = sfmap[2]
        nofm = sfmap[3]
        tmp = int(hofm * wofm / self.h_pe) + math.ceil((hofm * wofm % self.h_pe) / self.h_pe)
        utl_m = hofm * wofm / self.h_pe / tmp
        # utl_m = 1
        tmp = int(nofm / self.w_pe) + math.ceil((nofm % self.w_pe) / self.w_pe)
        utl_n = nofm / self.w_pe / tmp
        tmp = int(nifm / self.h_pe) + math.ceil((nifm % self.h_pe) / self.h_pe)
        utl_k = nifm / self.h_pe / tmp
        return (utl_m, utl_n, utl_k)

    def vecu_utl(self, sfmap):
        '''
        :param sfmap: shape of feature map both IFM, OFM, (hofm, wofm, nifm, nofm)
        # 1 dimensions utilization in Vector unit, in each round:
        :return: the utilization in 3 dimensions (1, nofm/svecu, 1)
        '''
        hofm = sfmap[0]
        wofm = sfmap[1]
        nifm = sfmap[2]
        nofm = sfmap[3]
        tmp = int(nofm / self.svecu) + math.ceil((nofm % self.svecu) / self.svecu)
        utl_n = nofm / self.svecu / tmp
        return (1, utl_n, 1)


class Cluster(AccCore):
    def __init__(self, scluster, smtxu, svecu, mtxu_cost, vecu_cost, core_mem: Core_mem_hierarchy):
        '''
        the cluster of multi-core DNN accelerators
        :param scluster:  the shape of cluster (xx x yy) AccCore
        :param smtxu: size of matrix unit (Systolic array nxn, e.g. 32x32) in accCore
        :param svecu: size of vector unit (Considering bw of l1 buf, usually) in accCore
        :param core_mem: memory hierarchy in accCore (ubuf, l1c, l0a/b/c)
        '''
        super(Cluster, self).__init__(smtxu, svecu, mtxu_cost, vecu_cost, core_mem)
        if isinstance(scluster, int):
            xlen = scluster
            ylen = scluster
        elif len(scluster) == 2:
            xlen, ylen = scluster[0], scluster[1]
        else:
            raise ValueError('Only support 2D X x Y cluster')

        ## record the ops for evaluation
        self.mtxu_ops = []
        self.mtxu_utl = []
        self.vecu_ops = []
        self.cycle = []
        self.core_buf_read = []
        self.core_buf_write = []
        self.xlen = xlen
        self.ylen = ylen
        self.core_full_mask = np.full((xlen,ylen), True)
        for i in range(self.xlen*self.ylen):
            buf_read = dict()
            buf_write = dict()
            self.mtxu_ops.append(0)
            self.vecu_ops.append(0)
            self.cycle.append(0)
            self.mtxu_utl.append([0, 0, 0, 0, 0])
            for buf_name in core_mem.memory.keys():
                buf_read[buf_name] = 0
                buf_write[buf_name] = 0
            self.core_buf_read.append(buf_read)
            self.core_buf_write.append(buf_write)


    def mtxu_ops_add(self, cidx, ops):
        '''
        accCore id execute ops MAC with matrix unit
        :param cid: 2D core idx in compute region
        :param ops:
        :return:
        '''
        idx = coreidx2idx(cidx, self.core_full_mask)
        self.mtxu_ops[idx] += ops

    def set_mtxu_utl(self, cidx, utl_list):
        '''
        record the mtxu ultilization of each core
        :param cidx:
        :param utl: [h_pe_utl, w_pe_utl, mtx_shape_utl],
        h_pe -> cin, w_pe -> cout, mtx_shape_utl -> h*w (the min highth of output matrix N is w_pe)
        :return:
        '''
        idx = coreidx2idx(cidx, self.core_full_mask)
        self.mtxu_utl[idx] = utl_list
    def vecu_ops_add(self, cidx, ops):
        idx = coreidx2idx(cidx, self.core_full_mask)
        self.vecu_ops[idx] += ops

    def set_cycle(self, cidx, cyc):
        idx = coreidx2idx(cidx, self.core_full_mask)
        tmp = self.cycle.copy()
        self.cycle[idx] = cyc
        # if pool:
        #     print(f'LRL cycle setting in core{idx}, cycle before {tmp}, after:{self.cycle}')

    def buf_read(self, cidx, buf_name, size):
        idx = coreidx2idx(cidx, self.core_full_mask)
        self.core_buf_read[idx][buf_name] += size
        # if buf_name == 'ubuf' and idx == 0:
        #     print(f'ubuf_read in core 0, size:{size}, total read: {self.core_buf_read[idx][buf_name]}')

    def buf_write(self, cidx, buf_name, size):
        idx = coreidx2idx(cidx, self.core_full_mask)
        self.core_buf_write[idx][buf_name] += size
        # if buf_name == 'ubuf' and idx == 0:
        #     print(f'ubuf_write in core 0, size:{size}, total write: {self.core_buf_write[idx][buf_name]}')

    def get_buf_bw(self, buf_name):
        return self.core_mem.get_buf_bw(buf_name)

    def get_buf_cost(self,buf_name):
        cost = 0
        for idx in range(self.xlen * self.ylen):
            cost += self.core_buf_read[idx][buf_name] * self.core_mem.get_buf_rcost(buf_name)
            cost += self.core_buf_write[idx][buf_name] * self.core_mem.get_buf_wcost(buf_name)
        return cost

    def get_tot_buf_cost(self):
        cost = 0
        for buf_name in self.core_mem.memory.keys():
            cost += self.get_buf_cost(buf_name)
        return cost

    def get_tot_cycle(self):
        return max(self.cycle)

    def get_mtxu_cost(self):
        return sum(self.mtxu_ops) * self.mtxu_cost

    def get_vecu_cost(self):
        return sum(self.vecu_ops) * self.vecu_cost

    def gen_core_info(self):
        mtxu_cost = np.array(self.mtxu_ops) * self.mtxu_cost
        vecu_cost = np.array(self.vecu_ops) * self.vecu_cost
        info = np.vstack((mtxu_cost, vecu_cost)).T
        for key in self.core_buf_read[0].keys():
            buf_info = []
            for i in range(len(self.core_buf_read)):
                cost = self.core_buf_read[i][key] * self.core_mem.get_buf_rcost(key) + self.core_buf_write[i][key] * self.core_mem.get_buf_wcost(key)
                buf_info.append(cost)
            info = np.column_stack((info, buf_info))
        return info

    def set_mtxu_ops_cost(self, cost):
        self.mtxu_cost = cost

    def clear(self):
        self.mtxu_ops.clear()
        self.vecu_ops.clear()
        self.mtxu_utl.clear()
        # print(f'before clear, cycle: {self.cycle}')
        self.core_buf_read.clear()
        self.core_buf_write.clear()
        self.cycle.clear()

        for i in range(self.xlen*self.ylen):
            buf_read = dict()
            buf_write = dict()
            self.cycle.append(0)
            self.mtxu_ops.append(0)
            self.vecu_ops.append(0)
            self.mtxu_utl.append([0, 0, 0, 0, 0])
            for buf_name in self.core_mem.memory.keys():
                buf_read[buf_name] = 0
                buf_write[buf_name] = 0
            self.core_buf_read.append(buf_read)
            self.core_buf_write.append(buf_write)

