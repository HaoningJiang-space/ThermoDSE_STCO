from .bufmanage import bufManager

class Nop():
    '''
    Network on Package model to calculate the NoP cost, Assuming that there are two model: uni-cast and multi-cast
    Currently, we don't consider the DRAM access from different ids/channels, assuming data will read/write from/to all DRAMs averagely
    since the column 0 and x+1 is the DRAM region, but in scheduler and evaluater the coreidx is only the compute region,
    we must transfer the coreidx in compute region into cluster region
    '''
    RIGHT=0
    DOWN =1
    LEFT =2
    UP   =3
    def __init__(self, nop_bw, noc_bw, dram_bw, dram_list, nop_hop_cost, noc_hop_cost, DRAM_acc_cost, shape, xcut, ycut):
        self.nop_bw = nop_bw
        self.noc_bw = noc_bw
        self.dram_bw = dram_bw
        self.tot_noc_hops = 0
        self.tot_nop_hops = 0
        self.tot_DRAM_access = 0
        self.nop_hop_cost = nop_hop_cost
        self.noc_hop_cost = noc_hop_cost
        self.DRAM_acc_cost = DRAM_acc_cost
        self.dram_list = dram_list
        self.shape = shape
        self.link_hops = [0] * shape[1] * (shape[0] + 2) * 4  # each core has 4 links for 4 dirctions (0: right, 1: down, 2: left, 3: up)
        self.xcut = xcut
        self.ycut = ycut
        self.x_step = int(self.shape[0]/xcut)
        self.y_step =int(self.shape[1]/ycut)

        self.nop_link_idx = []
        for y in range(self.shape[1]):
            self.nop_link_idx.append(self.get_link_idx(0,y,self.RIGHT)) 
            self.nop_link_idx.append(self.get_link_idx(1,y,self.LEFT))
            self.nop_link_idx.append(self.get_link_idx(shape[0]  ,y, self.RIGHT))
            self.nop_link_idx.append(self.get_link_idx(shape[0]+1,y, self.LEFT))
        if xcut > 1:
            x_step = int(self.shape[0]/xcut)
            for x in range(x_step, self.shape[0], x_step):
                for y in range(self.shape[1]):
                    self.nop_link_idx.append(self.get_link_idx(x-1+1,y, self.RIGHT )) # +1 because the first row is DRAM PHY DIE
                    self.nop_link_idx.append(self.get_link_idx(x-1+2,y, self.LEFT )) 
        if ycut > 1:
            for y in range(self.y_step, self.shape[1], self.y_step):
                for x in range(self.shape[0]):
                    self.nop_link_idx.append(self.get_link_idx(x+1, y-1  , self.DOWN))
                    self.nop_link_idx.append(self.get_link_idx(x+1, y-1+1, self.UP))

    def read_from_DRAM(self, dst, volume):
        cluster_idx = []
        for dst_idx in dst:
            cluster_idx.append(self.cidx2clusteridx(dst_idx))
        if len(dst) == 1:
            self.unicast_dram(cluster_idx[0], volume)
        else:
            self.multicast_dram(cluster_idx, volume)
        # if self.tot_nop_hops != self.tot_DRAM_access:
        #     print(f'Check: read from DRAM, but nop calculation wrong, the core dst is {dst}')
        #     if len(dst) == 1:
        #         self.unicast_dram(cluster_idx[0], volume)
        #     else:
        #         self.multicast_dram(cluster_idx, volume)

    def write_to_DRAM(self, src, volume):
        src_clt_idx = self.cidx2clusteridx(src)
        self.unicast_to_dram(src_clt_idx, volume)
        # if self.tot_nop_hops != self.tot_DRAM_access:
        #     print(f'Check: write into DRAM, but nop calculation wrong, the src is {src}')
        #     self.unicast_to_dram(src_clt_idx, volume)


    def move_between_core(self, src, dst, volume):
        src_clt_idx = self.cidx2clusteridx(src)
        dst_clt_idx = []
        for dst_idx in dst:
            dst_clt_idx.append(self.cidx2clusteridx(dst_idx))
        if len(dst) == 1:
            self.unicast(src_clt_idx, dst_clt_idx[0], volume)
        else:
            self.multicast(src_clt_idx, dst_clt_idx, volume)

    def unicast_dram(self, dst, size, dram_id=None):
        dram_len = len(self.dram_list)
        size_each = size / dram_len
        for dram in self.dram_list:
            self.unicast(dram, dst, size_each)  ## NoP hops update
        self.tot_DRAM_access += size   #DRAM access update

    def unicast_to_dram(self, src, size, dram_id = None):
        dram_len = len(self.dram_list)
        size_each = size / dram_len
        for dram in self.dram_list:
            self.unicast(src, dram, size_each)  ## NoP hops update
        self.tot_DRAM_access += size  # DRAM access update

    def multicast_dram(self, dst_list, size, dram_id=None):
        # multi-core read the same data from dram
        dram_len = len(self.dram_list)
        size_each = size / dram_len
        for dram in self.dram_list:
            self._multicastCalc(dram, dst_list, size_each)
        self.tot_DRAM_access += size

    def unicast(self,src_cidx, dst_cidx, size):
        tot_hops = self._unicastCalc(src_cidx, dst_cidx, size)
        nop_hops = self.NoP_link_calc(src_cidx, dst_cidx) * size
        if tot_hops < nop_hops:
            tot_hops = self._unicastCalc(src_cidx, dst_cidx, size)
            nop_hops = self.NoP_link_calc(src_cidx, dst_cidx) * size
            raise ValueError(f'tot_hops:{tot_hops} > nop hopts {nop_hops}, src: {src_cidx},dst: {dst_cidx}')
        self.tot_noc_hops += tot_hops - nop_hops
        self.tot_nop_hops += nop_hops

    def multicast(self, src_cidx, dst_cidx_list, size):
        self.tot_noc_hops += self._multicastCalc(src_cidx, dst_cidx_list, size)

    def clear(self):
        self.tot_noc_hops = 0
        self.tot_nop_hops = 0
        self.tot_DRAM_access = 0
        self.link_hops = [0] * len(self.link_hops)

    def _multicastCalc(self, src_cidx, dst_cidx_list, size):
        '''
        multi-cast calculation from src_cidx to all dst_cidx
        (this is a approximated value not accute value, since this question is NP-hard, same as SET/Gemini)
        :param src_cidx: (x0, y0)
        :param dst_cidx_list: [(x1, y1), ..., (xn,yn)]
        :param size: size of this data range
        :return:
        '''
        src_x = src_cidx[0]
        src_y = src_cidx[1]
        h = 0
        dst_len = len(dst_cidx_list)
        # stored dst_cidx_list along x dimension, and get the hops in x dim
        dst_cidx_list = sorted(dst_cidx_list, key=lambda x: x[0])
        for x in range(src_x, dst_cidx_list[0][0], -1):
            link_idx = self.get_link_idx(x, src_y, 2)
            self.link_hops[link_idx] += size
        for x in range(src_x, dst_cidx_list[-1][0], 1):
            link_idx = self.get_link_idx(x, src_y, 0)
            self.link_hops[link_idx] += size
        h += max(src_x, dst_cidx_list[-1][0]) - min(src_x, dst_cidx_list[0][0])
        h_p_temp = self.NoP_link_calc_dir(max(src_x, dst_cidx_list[-1][0]), min(src_x, dst_cidx_list[0][0]), isX=True)
        h -= h_p_temp
        self.tot_nop_hops += h_p_temp * size
        # caculate hops in y dim, sweep the dst core in same x column, h = max(cur_y, max_y) - min(src_y, min_y)
        cur_x = dst_cidx_list[0][0]
        min_y = dst_cidx_list[0][1]
        for i, dst_cidx in enumerate(dst_cidx_list[1:]):
            tmp = i + 1
            if tmp < dst_len and dst_cidx[0] == cur_x: continue
            for y in range(src_y, min_y, -1):
                link_idx = self.get_link_idx(cur_x, y, 1)
                self.link_hops[link_idx] += size
            for y in range(src_y, dst_cidx_list[i][1], 1):
                link_idx = self.get_link_idx(cur_x, y, 3)
                self.link_hops[link_idx] += size
            h += max(src_y, dst_cidx_list[i][1]) - min(src_y, min_y)
            h_p_temp = self.NoP_link_calc_dir(max(src_y, dst_cidx_list[i][1]), min(src_y, min_y), isX=False)
            h -= h_p_temp
            self.tot_nop_hops += h_p_temp * size
            if tmp == dst_len: break
            cur_x = dst_cidx_list[tmp][0]
            min_y = dst_cidx_list[tmp][1]
        return h * size

    def _unicastCalc(self, src_cidx, dst_cidx, size):
        x_dir = 0 if (dst_cidx[0] > src_cidx[0]) else 2
        y_dir = 3 if (dst_cidx[1] > src_cidx[1]) else 1
        dx = 1 if (dst_cidx[0] > src_cidx[0]) else -1
        dy = 1 if (dst_cidx[1] > src_cidx[1]) else -1
        for x in range(src_cidx[0], dst_cidx[0], dx):
            link_idx = self.get_link_idx(x, src_cidx[1], x_dir)
            self.link_hops[link_idx] += size
        for y in range(src_cidx[1], dst_cidx[1], dy):
            link_idx = self.get_link_idx(dst_cidx[0], y, y_dir)
            self.link_hops[link_idx] += size
        return size * (abs(src_cidx[0] - dst_cidx[0]) + abs(src_cidx[1] - dst_cidx[1]))
    
    def get_link_idx(self, x, y, dir):
        return (y * self.shape[0] + x) * 4 + dir

    def get_tot_noc_hops(self):
        return self.tot_noc_hops

    def get_tot_nop_hops(self):
        nop_hops = self.tot_nop_hops - self.tot_DRAM_access
        assert nop_hops >= 0, f'nop_hops : {self.tot_nop_hops}, DRAM_access : {self.tot_DRAM_access} '
        return nop_hops

    def get_tot_DRAM_access(self):
        return self.tot_DRAM_access

    def get_noc_cost(self):
        return self.tot_noc_hops * self.noc_hop_cost

    def get_nop_cost(self):
        return self.get_tot_nop_hops() * self.nop_hop_cost

    def get_dram_cost(self):
        return self.tot_DRAM_access * self.DRAM_acc_cost


    def get_cost(self):
        return self.get_noc_cost() + self.get_nop_cost() + self.get_dram_cost()

    def get_nocp_time(self):
        nop_hop_max = 0
        noc_hop_max = 0
        for idx, hops in enumerate(self.link_hops):
            if idx in self.nop_link_idx:
                nop_hop_max = max(hops, nop_hop_max)
            else:
                noc_hop_max = max(hops, noc_hop_max)
        return max(nop_hop_max// self.nop_bw, noc_hop_max//self.noc_bw)

    def get_DRAM_time(self):
        return self.tot_DRAM_access // (len(self.dram_list) * self.dram_bw)

    def get_time(self):
        nop_time = self.get_nocp_time()
        dram_time = self.get_DRAM_time()
        return max(nop_time, dram_time)

    def cidx2clusteridx(self, cidx):
        return (cidx[0] + 1, cidx[1])
    
    def NoP_link_calc(self, src, dst):
        x_max = max(src[0], dst[0])
        x_min = min(src[0], dst[0])
        x_NoP_hop = 0
        y_NoP_hop =0
        # caculate the DRAM link first
        if (x_min == 0 and x_max >0):   ## only DRAM in x==0 and x
            x_NoP_hop +=1
            x_min += 1
        if(x_max == self.shape[0] + 1 and x_min < x_max):
            x_NoP_hop +=1
            x_max -= 1
        if (x_max-x_min) >= self.x_step:  # since for x direction, core start from 1 to self.shape[0] + 1
            x_nearest_interval = x_max - x_max % self.x_step
            assert x_nearest_interval > 0
            if x_nearest_interval > x_min:
                x_NoP_hop += 1 + int((x_nearest_interval - 1 - x_min) / self.x_step)
        
        y_max = max(src[1], dst[1])
        y_min = min(src[1], dst[1])
        if (y_max-y_min) >= self.y_step:   # for y direction, core start from 0 to self.shape[1]
            y_nearest_interval = y_max - y_max % self.y_step
            assert y_nearest_interval > 0
            if y_nearest_interval > y_min:
                y_NoP_hop = 1 + int((y_nearest_interval - 1 - y_min) / self.y_step)
         
        return x_NoP_hop + y_NoP_hop    

    def NoP_link_calc_dir(self, x1, x2, isX:bool):
        step = self.x_step if isX else self.y_step
        x_max = max(x1,x2)
        x_min = min(x1,x2)
        x_NoP_hop = 0
        if isX and x_min ==0 and x_max >0:
            x_NoP_hop += 1  ## X == 0 only have DRAM
            x_min += 1
        if x_max == self.shape[0] + 1 and x_min < x_max:
            x_NoP_hop += 1
            x_max -= 1
        if (x_max - x_min) >= step:
            x_nearest_interval = x_max - x_max % step
            assert x_nearest_interval > 0
            if x_nearest_interval > x_min:
                x_NoP_hop += 1 + int((x_nearest_interval - 1 - x_min) / step)
        return x_NoP_hop
    
    def __str__(self):
        str_ = 'total hops: ' + str(self.tot_hops) + '. total DRAM access: ' + str(self.tot_DRAM_access)+'\n'
        return str_

