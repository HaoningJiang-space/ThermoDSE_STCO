import math
import os

import numpy as np
from matplotlib import pyplot as plt
from .util import coreidx2idx,  idx2coreidx

class Statistic:
    MTXU = 0
    VECU = 1
    UBUF = 2
    L0A = 3
    L0B = 4
    L0C = 5
    L1C = 6
    NAME_LIST = ['mtxu', 'vecu', 'ubuf', 'l0a', 'l0b', 'l0c', 'l1c']
    NAME_LIST_3D = ['mtxu', 'vecu', 'ubuf', 'ibuf', 'obuf', 'io_0', 'io_1', 'io_2', 'io_3']
    def __init__(self, scluster,schiplet, mtxu_shape, grid_shape, clk_freq):
        '''
        this is a monitor to statistic the latency energy cost
        :param 
        scluster: (xlen, ylen), shape of accelerator core
        schiplet: shape of chiplet for this cluster
        energy_dict -> key: nn_name, value: a array of cost when executing this nn
        '''
        self.cxlen = scluster[0]
        self.cylen = scluster[1]
        self.xcut = schiplet[0]
        self.ycut = schiplet[1]
        self.clk_freq = clk_freq
        self.core_dict = dict()
        self.latency_dict = dict()
        self.noc_dict = dict()
        self.nop_dict = dict()
        self.dram_dict = dict()
        self.core_utl_dict = dict()
        self.mtxu_utl_dict = dict()
        self.core_full_mask = np.full((self.cxlen, self.cylen), True)

        assert mtxu_shape[0] % grid_shape[0] == 0 and mtxu_shape[1] % grid_shape[1] == 0, (f'please split matrix unit '
                                                                                           f'uniformly')
        self.grid_xlen = mtxu_shape[0] // grid_shape[0]
        self.grid_ylen = mtxu_shape[1] // grid_shape[1]

    def init_exe_info(self, nn_name, tot_order):
        '''
        initial the dict for the application nn_name. we split the matrix unit into multiple grids, which can observe
        the thermal aggregation phenomenon.
        :param nn_name:
        :param tot_order:
        :param mtxu_shape: the shape of matrix unit,
        :param grid_shape: the shape of grid unit
        :return:
        '''
        cnt = 0
        name = nn_name
        if name in self.core_dict.keys():
            while name in self.core_dict.keys():
                cnt +=1
                name = nn_name + str(cnt)

        self.core_dict[name] = np.zeros((tot_order, self.cxlen * self.cylen, 7))
        self.latency_dict[name] = np.zeros(tot_order)
        self.nop_dict[name] = np.zeros(tot_order)
        self.noc_dict[name] = np.zeros(tot_order)
        self.dram_dict[name] = np.zeros(tot_order)
        self.core_utl_dict[name] = np.zeros((tot_order, self.cxlen * self.cylen))
        self.mtxu_utl_dict[name] = np.zeros((tot_order, self.cxlen * self.cylen, self.grid_xlen, self.grid_ylen))
        return name
    
    def cost_times(self,nn_name, times):
        self.core_dict[nn_name] *= times
        self.latency_dict[nn_name] *= times
        self.nop_dict[nn_name] *= times
        self.noc_dict[nn_name] *= times
        self.dram_dict[nn_name] *= times

    def cost_copy(self):
        for nn_name in self.core_dict.keys():
            for i in range(1, self.cxlen * self.cylen):
                self.core_dict[nn_name][:,i,:] = self.core_dict[nn_name][:, 0, :]
                self.core_utl_dict[nn_name][:,i] = self.core_utl_dict[nn_name][:,0]
                self.mtxu_utl_dict[nn_name][:,i] = self.mtxu_utl_dict[nn_name][:,0]


    def update_external_info(self, nn_name, cur_order, latency,noc_cost, nop_cost, dram_cost):
        self.latency_dict[nn_name][cur_order] = latency
        self.noc_dict[nn_name][cur_order] = noc_cost
        self.nop_dict[nn_name][cur_order] = nop_cost
        self.dram_dict[nn_name][cur_order] = dram_cost

    def update_internal_info(self, nn_name, cur_order, core_info, core_utl):
        self.core_dict[nn_name][cur_order, :, :] = core_info
        self.core_utl_dict[nn_name][cur_order, :] = core_utl

    def update_mtxu_grid_utl(self, nn_name, cur_order, mtxu_utl_info):
        '''
        record the grid utl from the mtxu_utl_info, 6 dims
        0 -> coreidx
        1 -> cout_loop_ideal, this is a float data, e.g. 4.1 means the after 4 times loops in cout,
            the 5th loop only 10% PEs in mtxu are working
        2 -> cout_loop_real, INT data, math.ceil(cout_loop_ideal)
        3 -> cin_loop_ideal, this is a float data, similar to cout_loop_ideal
        4 -> cin_loop_real, INT data, math.ceil(cin_loop_ideal)
        5 -> hw_utl, since the minimum size of matrix unit is (w_pe (N), h_pe (K)) x (h_pe (K), w_pe (M)), when N < w_pe,
            one of input data should stall to wait the other data updating. N in CONV is determined by ho and wo
        :param mtxu_utl_info:
        :return:
        '''
        for cidx in range(len(mtxu_utl_info)):
            ult_info = mtxu_utl_info[cidx]
            co_loop_ideal = ult_info[0]
            co_loop_real = ult_info[1]
            cin_loop_ideal = ult_info[2]
            cin_loop_real = ult_info[3]
            utl_array = np.zeros((self.grid_xlen, self.grid_ylen))
            co_full_ult = (co_loop_real == co_loop_ideal)
            cin_full_ult = (cin_loop_real == cin_loop_ideal)
            if co_full_ult and cin_full_ult:
                utl_array[:, :] = co_loop_real * cin_loop_ideal
            elif co_full_ult and not cin_full_ult:
                cin_full_end = math.ceil((cin_loop_ideal + 1 - cin_loop_real) * self.grid_ylen)
                utl_array[ :, 0:cin_full_end] = co_loop_real * cin_loop_real
                utl_array[ :, cin_full_end: ] = co_loop_real * math.floor(cin_loop_ideal)
            elif not co_full_ult and cin_full_ult:
                co_full_ult = math.ceil((co_loop_ideal + 1 - co_loop_real) * self.grid_xlen)
                utl_array[0: co_full_ult, :] = cin_loop_real * co_loop_real
                utl_array[co_full_ult : , :] = cin_loop_real * math.floor(co_loop_ideal)
            else:
                cin_full_end = math.ceil((cin_loop_ideal + 1 - cin_loop_real) * self.grid_ylen)
                co_full_end = math.ceil((co_loop_ideal + 1 - co_loop_real) * self.grid_xlen)
                utl_array[0: co_full_end, 0: cin_full_end] = co_loop_real * cin_loop_real
                utl_array[co_full_end: , 0: cin_full_end] = cin_loop_real * math.floor(co_loop_ideal)
                utl_array[0: co_full_end, cin_full_end :] = co_loop_real * math.floor(cin_loop_ideal)
                utl_array[co_full_end:, cin_full_end : ] = math.floor(cin_loop_ideal) * math.floor(co_loop_ideal)
            utl_array = utl_array / (co_loop_real * cin_loop_real)
            self.mtxu_utl_dict[nn_name][cur_order, cidx, :, :] = utl_array.copy()

    def draw_core_utl(self, core=None):
        if core is None:
            isallcore = True
        else:
            isallcore = False
        net_cnt = 0
        for net, core_utl in self.core_utl_dict.items():
            tot_order = core_utl.shape[0]
            x = np.arange(tot_order)
            plt.figure(net_cnt)
            if isallcore:
                for i in range(self.cxlen * self.cylen):
                    plt.plot(x, core_utl[:, i], label=f'core{i}')
            else:
                for i in core:
                    plt.plot(x, core_utl[:, i], label=f'core{i}')
            plt.title(f'For {net}, core utilization')
            plt.xlabel('execution order')
            plt.ylabel('utilization')
            plt.legend()
            plt.grid()
            net_cnt += 1
        plt.show()

    def draw_mtxu_utl(self, core=None):
        if core is None:
            isallcore = True
        else:
            isallcore = False
        net_cnt = 0
        for net, mtxu_utl in self.mtxu_utl_dict.items():
            tot_order = mtxu_utl.shape[0]
            x = np.arange(tot_order)
            plt.figure(net_cnt)
            print(mtxu_utl.shape)
            if isallcore:
                for i in range(self.cxlen * self.cylen):
                    for j in range(self.grid_xlen):
                        for k in range(self.grid_ylen):
                            plt.plot(x, mtxu_utl[:, i, j, k], label=f'core{i}.grid ({j},{k})')
            else:
                for i in core:
                    for j in range(self.grid_xlen):
                        for k in range(self.grid_ylen):
                            plt.plot(x, mtxu_utl[:, i, j, k], label=f'core{i}.grid ({j},{k})')

            plt.title(f'For {net}, matrx unit utilization')
            plt.xlabel('execution order')
            plt.ylabel('utilization')
            plt.legend()
            plt.grid()
            net_cnt += 1
        plt.show()
    
    def get_nn_cost(self, nn_name):
        breakdown = self.get_nn_breakdown(nn_name)
        return breakdown['latency_cycles'], breakdown['modeled_energy_pj_excluding_compute']

    def get_nn_breakdown(self, nn_name):
        """Return a JSON-safe per-workload latency/energy snapshot.

        The historical optimization metric returned by :meth:`get_nn_cost`
        excludes MTXU/VECU compute energy.  Export both that exact boundary
        and the inclusive component total so downstream STCO code cannot
        accidentally label the legacy quantity as generic total energy.
        Values are raw simulator pJ/cycles; unit conversion belongs to the
        caller that aggregates workloads.
        """
        if nn_name not in self.latency_dict:
            raise KeyError(f'unknown workload execution name: {nn_name}')

        core = self.core_dict[nn_name]
        energy_pj = {
            'nop': float(np.sum(self.nop_dict[nn_name])),
            'noc': float(np.sum(self.noc_dict[nn_name])),
            'dram': float(np.sum(self.dram_dict[nn_name])),
            'compute': float(np.sum(core[:, :, self.MTXU]) + np.sum(core[:, :, self.VECU])),
            'ubuf': float(np.sum(core[:, :, self.UBUF])),
            'input_buffers': float(np.sum(core[:, :, self.L0A]) + np.sum(core[:, :, self.L0B])),
            'output_buffers': float(np.sum(core[:, :, self.L0C]) + np.sum(core[:, :, self.L1C])),
        }
        total_including_compute = sum(energy_pj.values())
        modeled_excluding_compute = total_including_compute - energy_pj['compute']
        return {
            'latency_cycles': float(np.sum(self.latency_dict[nn_name])),
            'energy_pj': energy_pj,
            'modeled_energy_pj_excluding_compute': float(modeled_excluding_compute),
            'total_energy_pj_including_compute': float(total_including_compute),
        }
    
    def gen_ptrace(self, nn_name, gen_path='../tmp/ptrace'):
        latency = np.sum(self.latency_dict[nn_name])/self.clk_freq
        file_name = os.path.join(gen_path,nn_name + '.ptrace')
        # generate name as header of ptrace file
        # str_= 'interposer\t'
        str_ = ''
        for name in self.NAME_LIST:
            for j in range(self.cylen):
                for i in range(self.cxlen):
                    str_ += name + '_' + str(j*self.cylen + i) + '\t'
        # generate the power trace
        str_ += '\n'
        # p_itp = (np.sum(self.nop_dict[nn_name]) + np.sum(self.dram_dict[nn_name])) * 1e-12 / latency
        # str_ += f'{p_itp:.4f}\t'
        for m in range(len(self.NAME_LIST)):
            for j in range(self.cylen):
                for i in range(self.cxlen):
                    idx = coreidx2idx((i,j), self.core_full_mask)
                    avg_p = np.sum(self.core_dict[nn_name][:, idx, m]) * 1e-12 / latency   # energy cost (pJ), latency (ns)
                    if 'mtxu' in self.NAME_LIST[m]:
                        avg_p = 1 * avg_p
                    str_ += f'{avg_p:.4f}\t'
        str_ += '\n'
        ptrace_file = open(file_name,'w')
        ptrace_file.write(str_)       
        ptrace_file.close() 
    
    def gen_ptrace_3D(self, nn_name, ics, gen_path='../tmp/ptrace'):
        latency = np.sum(self.latency_dict[nn_name])/self.clk_freq
        file_name = os.path.join(gen_path,nn_name + '_3D.ptrace')
        # generate name as header of ptrace file
        str_= 'interposer\t'
        # str_ = ''
        for name in self.NAME_LIST_3D:
            for j in range(self.cylen):
                for i in range(self.cxlen):
                    str_ += name + '_' + str(j*self.cxlen + i) + '\t'
        if ics > 1e-6:            
            for j in range(self.cylen):
                for i in range(self.cxlen):
                    idx = j * self.cxlen + i
                    if i < self.cxlen - 1:
                        str_ += 'blockX_{}\t'.format(idx)
                    if j <self.cylen - 1:
                        str_ += 'blockY_{}\t'.format(idx)
                    if i < self.cxlen - 1 and j < self.cylen - 1:
                        str_+= 'blockXY_{}\t'.format(idx)
        # str_ += 'blk_l\tblk_t'
        # generate the power trace
        str_ += '\n'
        p_itp = (np.sum(self.nop_dict[nn_name]) + np.sum(self.dram_dict[nn_name])) * 1e-12 / latency
        str_ += f'{p_itp:.4f}\t'
        for m in range(len(self.NAME_LIST_3D)):
            for j in range(self.cylen):
                for i in range(self.cxlen):
                    idx = coreidx2idx((i,j), self.core_full_mask)
                    if 'mtxu' in self.NAME_LIST_3D[m] or 'vecu' in self.NAME_LIST_3D[m] or 'ubuf' in self.NAME_LIST_3D[m]:
                        avg_p = np.sum(self.core_dict[nn_name][:, idx, m]) * 1e-12 / latency   # energy cost (pJ), latency (ns)
                    elif 'ibuf' in self.NAME_LIST_3D[m]:
                        avg_p = (np.sum(self.core_dict[nn_name][:, idx, self.L0A]) + np.sum(self.core_dict[nn_name][:, idx, self.L0B])) * 1e-12 / latency
                    elif 'obuf' in self.NAME_LIST_3D[m]:
                        avg_p = (np.sum(self.core_dict[nn_name][:, idx, self.L0C]) + np.sum(self.core_dict[nn_name][:, idx, self.L1C])) * 1e-12 / latency
                    elif 'io' in self.NAME_LIST_3D[m]:
                        avg_p = 0
                    else:
                        raise TypeError(f'This compnent is not supported {self.NAME_LIST_3D[m]}')
                    str_ += f'{avg_p:.4f}\t'
        if ics > 1e-6:
            for j in range(self.cylen):
                for i in range(self.cxlen):
                    avg_p = 0
                    if i < self.cxlen - 1:
                        str_ += f'{avg_p:.4f}\t'
                    if j <self.cylen - 1:
                        str_ += f'{avg_p:.4f}\t'
                    if i < self.cxlen - 1 and j < self.cylen - 1:
                        str_ += f'{avg_p:.4f}\t'
        # str_ += f'{0:.4f}\t{0:.4f}'
        str_ += '\n'
        ptrace_file = open(file_name,'w')
        ptrace_file.write(str_)       
        ptrace_file.close() 
                
                
    def gen_all_ptrace_3D(self, isRunBaseline3 = False, gen_path='../tmp/ptrace'):
        # latency = np.sum(self.latency_dict[nn_name])/self.clk_freq
        latency = 0
        max_latency = 0
        for cycle_order in self.latency_dict.values():
            latency_tmp = np.sum(cycle_order) / self.clk_freq
            latency += latency_tmp
            if latency_tmp > max_latency:
                max_latency = latency_tmp
            # if tmp >  latency:
            #     latency = tmp
        # print(latency)
        if isRunBaseline3: latency = max_latency * len(self.latency_dict)  # consider the syn latency
        file_name = os.path.join(gen_path,'cores_3D.ptrace')
        # generate name as header of ptrace file
        str_= 'interposer\tinterposer_e0\tinterposer_e1\tinterposer_e2\tinterposer_e3\t'
        # str_ = ''
        for name in self.NAME_LIST_3D:
            for j in range(self.cylen):
                for i in range(self.cxlen):
                    str_ += name + '_' + str(j*self.cxlen + i) + '\t'
    
        for j in range(self.cylen):
            for i in range(self.cxlen):
                idx = j * self.cxlen + i
                if i < self.cxlen - 1:
                    str_ += 'blockX_{}\t'.format(idx)
                if j <self.cylen - 1:
                    str_ += 'blockY_{}\t'.format(idx)
                if i < self.cxlen - 1 and j < self.cylen - 1:
                    str_+= 'blockXY_{}\t'.format(idx)
        str_ += 'eblk0\teblk1\teblk2\teblk3\t'
        # str_ += 'dram\tdram_e0\tdram_e1\tdram_e2\tdram_e3\t'
        # str_ += 'blk_l\tblk_t'
        # generate the power trace
        str_ += '\n'
        p_itp = 0
        p_noc = 0
        for nn_name in self.nop_dict.keys():
            p_itp += (np.sum(self.nop_dict[nn_name])) * 1e-12 / latency
            p_noc +=  np.sum(self.noc_dict[nn_name]) * 1e-12 / latency
        str_ += f'{p_itp:.4f}\t{0:.4f}\t{0:.4f}\t{0:.4f}\t{0:.4f}\t'
        for m in range(len(self.NAME_LIST_3D)):
            for j in range(self.cylen):
                for i in range(self.cxlen):
                    avg_p = 0
                    for nn_name in self.core_dict.keys():
                        idx = coreidx2idx((i,j), self.core_full_mask)
                        if 'mtxu' in self.NAME_LIST_3D[m] or 'vecu' in self.NAME_LIST_3D[m] or 'ubuf' in self.NAME_LIST_3D[m]:
                            avg_p += np.sum(self.core_dict[nn_name][:, idx, m]) * 1e-12 / latency   # energy cost (pJ), latency (ns)
                        elif 'ibuf' in self.NAME_LIST_3D[m]:
                            avg_p += (np.sum(self.core_dict[nn_name][:, idx, self.L0A]) + np.sum(self.core_dict[nn_name][:, idx, self.L0B])) * 1e-12 / latency
                        elif 'obuf' in self.NAME_LIST_3D[m]:
                            avg_p += (np.sum(self.core_dict[nn_name][:, idx, self.L0C]) + np.sum(self.core_dict[nn_name][:, idx, self.L1C])) * 1e-12 / latency
                        elif 'io' in self.NAME_LIST_3D[m]:
                            if p_noc > 0:
                                avg_p = p_noc / ((self.cylen - 1) * 2 *self.cxlen + (self.cxlen - 1) * 2 *self.cylen)
                            else:
                                avg_p = 0
                        else:
                            raise TypeError(f'This compnent is not supported {self.NAME_LIST_3D[m]}')
                    str_ += f'{avg_p:.4f}\t'
        for j in range(self.cylen):
            for i in range(self.cxlen):
                avg_p = 0
                if i < self.cxlen - 1:
                    str_ += f'{avg_p:.4f}\t'
                if j <self.cylen - 1:
                    str_ += f'{avg_p:.4f}\t'
                if i < self.cxlen - 1 and j < self.cylen - 1:
                    str_ += f'{avg_p:.4f}\t'
        str_ += f'{0:.4f}\t{0:.4f}\t{0:.4f}\t{0:.4f}\t'
        p_itp = 0
        # for nn_name in self.nop_dict.keys():
        #     p_itp += np.sum(self.dram_dict[nn_name]) * 1e-12 / latency
        # # str_ += f'{0:.4f}\t{0:.4f}'
        # str_ += f'{p_itp:.4f}\t{0:.4f}\t{0:.4f}\t{0:.4f}\t{0:.4f}\t'
        str_ += '\n'
        ptrace_file = open(file_name,'w')
        ptrace_file.write(str_)       
        ptrace_file.close() 

    def report_core_power_var(self):
        latency = 0
        for nn_name in self.latency_dict.keys():
            tmp = np.sum(self.latency_dict[nn_name])/self.clk_freq
            if tmp > latency:
                latency = tmp
        for i in range(self.cxlen):
            for j in range(self.cylen):
                pwr = 0
                idx = coreidx2idx((i,j), self.core_full_mask)
                for nn_name in self.core_dict.keys():
                    pwr += np.sum(self.core_dict[nn_name][:, idx, :])* 1e-12 / latency
                print(f'averge power at core({i}, {j}):{pwr}')
    
    def report_energy_stats(self):
        noc_cost = 0
        nop_cost = 0
        comp_cost = 0
        ubuf_cost = 0
        ibuf_cost = 0
        obuf_cost = 0
        dram_cost = 0
        latency = 0
        for nn_name in self.latency_dict.keys():
            latency += np.sum(self.latency_dict[nn_name])/self.clk_freq
            nop_cost += (np.sum(self.nop_dict[nn_name]) ) 
            noc_cost += np.sum(self.noc_dict[nn_name])
            comp_cost += np.sum(self.core_dict[nn_name][:, :, self.MTXU]) + np.sum(self.core_dict[nn_name][:, :, self.VECU])
            ibuf_cost += (np.sum(self.core_dict[nn_name][:, :, self.L0A]) + np.sum(self.core_dict[nn_name][:, :, self.L0B]))
            ubuf_cost += np.sum(self.core_dict[nn_name][:, :, self.UBUF])
            obuf_cost += np.sum(self.core_dict[nn_name][:, :, self.L0C]) + np.sum(self.core_dict[nn_name][:, :, self.L1C])
            dram_cost += np.sum(self.dram_dict[nn_name])
        total_cost = nop_cost + noc_cost + comp_cost + ibuf_cost + ubuf_cost + obuf_cost + dram_cost
        print(f'Delay:{latency} s, Energy:{total_cost * 1e-6} uJ')
        print(f'Energy: noc:{int(noc_cost), int(noc_cost/total_cost*100)}%, nop:{int(nop_cost), int(nop_cost/total_cost*100)}%, comp:{int(comp_cost), int(comp_cost/total_cost*100)}%, ubuf:{int(ibuf_cost + obuf_cost + ubuf_cost), int((ibuf_cost + obuf_cost + ubuf_cost)/total_cost*100)}%')
        print(f'DRAM:{int(dram_cost)}, dram/total:{int(dram_cost/(dram_cost +total_cost) * 100)}%')
        print(f'Power: noc:{noc_cost * 1e-9/latency} mW, nop:{nop_cost* 1e-9/latency} mW, comp:{comp_cost * 1e-9/latency} mW, buf:{(ibuf_cost + obuf_cost + ubuf_cost) * 1e-9/ latency} mW')

    def report_network_stats(self):
        for nn_name in self.latency_dict.keys():
            # print(self.latency_dict[nn_name])
            latency = np.sum(self.latency_dict[nn_name])/self.clk_freq
            nop_cost = (np.sum(self.nop_dict[nn_name]) ) 
            noc_cost = np.sum(self.noc_dict[nn_name])
            comp_cost = np.sum(self.core_dict[nn_name][:, :, self.MTXU]) + np.sum(self.core_dict[nn_name][:, :, self.VECU])
            ibuf_cost = (np.sum(self.core_dict[nn_name][:, :, self.L0A]) + np.sum(self.core_dict[nn_name][:, :, self.L0B]))
            ubuf_cost = np.sum(self.core_dict[nn_name][:, :, self.UBUF])
            obuf_cost = np.sum(self.core_dict[nn_name][:, :, self.L0C]) + np.sum(self.core_dict[nn_name][:, :, self.L1C])
            dram_cost = np.sum(self.dram_dict[nn_name])
            total_cost = nop_cost + noc_cost + comp_cost + ibuf_cost + ubuf_cost + obuf_cost + dram_cost
            print(f'-------------------{nn_name} statstic-------------------------')
            print(f'Delay:{latency * 1e3} ms, Energy:{total_cost * 1e-9} mJ')
            print(f'Energy: nocp:{int(noc_cost+nop_cost)}, noc:{int(noc_cost), int(noc_cost/total_cost*100)}%, nop:{int(nop_cost), int(nop_cost/total_cost*100)}%,')
            print(f'comp:{int(comp_cost), int(comp_cost/total_cost*100)}%, ubuf:{int(ubuf_cost), int((ubuf_cost)/total_cost*100)}%,')
            print(f'iubf:{int(ibuf_cost), int(ibuf_cost/total_cost*100)}%, obuf:{int(obuf_cost), int(obuf_cost/total_cost*100)}%')
            print(f'DRAM: {int(dram_cost)}, dram/total:{int(dram_cost/(dram_cost +total_cost) * 100)}%')
            print(f'Power: noc:{noc_cost * 1e-9/latency} mW, nop:{nop_cost* 1e-9/latency} mW, comp:{comp_cost * 1e-9/latency} mW, buf:{(ibuf_cost + obuf_cost + ubuf_cost) * 1e-9/ latency} mW')


    def clear(self):
        self.core_dict.clear()
        self.latency_dict.clear()
        self.noc_dict.clear()
        self.nop_dict.clear()
        self.dram_dict.clear()
        self.core_utl_dict.clear()
        self.mtxu_utl_dict.clear()
