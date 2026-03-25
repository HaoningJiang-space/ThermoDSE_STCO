import random
from collections import OrderedDict

import numpy as np
import math 

from .taskdag import TaskDAG, CONVTaskNode, LRLTaskNode, GemmTaskNode, DPConvTaskNode
from .accCore import Cluster
from .util import idx2coreidx, manhattan_distance, find_coreidx_dist, coreidx2idx, eu_distance_sqrt, gen_map_order
from .bufmanage import bufManager


class Schedule():
    def __init__(self,core_mask, taskDag: TaskDAG, cluster: Cluster, baseline1=False, verbose=True):
        '''
        Schedule is to map task node onto each acc core and
        determine if the ofm should be written into DRAM
        '''
        # self.xlen = xlen
        # self.ylen = ylen
        self.cluster = cluster
        self.core_mask = core_mask
        self.numCore = np.sum(core_mask)
        self.ubuf = cluster.get_buf_size('ubuf')
        self.taskDag = taskDag
        self.exe_order = list()
        self.map_order = list()
        self.verbose = verbose
        self.baseline1 = baseline1    ## the baseline of chiplet-gym
        self.idx2coreidx_map = []   ## since the idx2coreidx is exhaustive method, do it only at init and cache them, 
                                    ## coreidx2idx -> self.idx2coreidx_map.index(coreidx), idx2coreidx -> self.idx2coreidx_map[idx]
        for i in range(self.numCore):
            self.idx2coreidx_map.append(idx2coreidx(i,self.core_mask))

    def schedule_dag(self, force_closed_core):
        task_list = list(self.taskDag.task_dict.keys())
        task_candidate = []
        depth_cnt = 0
        task_cur = []
        task_done = ['__INPUT_LAYER__']
        while len(task_list) > 0:
            # update candidate
            while len(task_candidate) < self.numCore and depth_cnt <= self.taskDag.depth:
                task_candidate += self.taskDag.task_dfs[depth_cnt]
                depth_cnt += 1
            task_nxt = self.select_task(force_closed_core, task_cur, task_candidate, task_done)
            self.exe_order.append(task_nxt.copy())
            task_cur = task_nxt.copy()
            task_done += task_nxt
            for task_done_tmp in task_cur:
                task_list.remove(task_done_tmp)
                task_candidate.remove(task_done_tmp)
                self.taskDag.set_task_exe_time(task_done_tmp,len(self.exe_order) - 1)

    def select_task(self, force_closed_core, task_cur: list, task_candidate: list, task_done: list):
        '''
        select a set of task for next round execution, currently communication-aware scheduling
        this is an initial version that can NOT fully utilize the data sharing in intra-layer level
        :param task_cur: the executing tasks in this round
        :param task_candidate: candidate tasks
        :param task_done: finished tasks
        :return: selected task for next round execution
        '''
        task_no_dep = []  ## select the task which all previous tasks have done
        num_task_sel = self.numCore - force_closed_core
        ## fliter out the task that can't execute in current round
        for task_cand in task_candidate:
            prevs = self.taskDag.prevs_dict[task_cand]
            prevs_done = True
            for prevs_tmp in prevs:
                if prevs_tmp not in task_done and prevs_tmp not in task_cur:
                    prevs_done = False
            if prevs_done:
                task_no_dep.append(task_cand)
        assert len(task_no_dep) > 0  # the scheduler is wrong, if can't get any task withouts dependency
        ## when candidate < #_CORE, all task can be executed in next round
        if len(task_no_dep) < num_task_sel:
            return task_no_dep
        elif len(task_cur) == 0 :
            # initial phase, not any task is executing
            # NOT a optimal selection considering buffer sharing within intra-layer level
            return task_no_dep[0:num_task_sel]
        else:
            ## find task which have maximum reused data volume
            data_reused_vol = []
            for i, task_core in enumerate(task_cur):
                reused_vol_tmp = dict()
                tasknode_exec = self.taskDag[task_core]
                for task_cand in task_no_dep:
                    tasknode_cand = self.taskDag[task_cand]
                    max_reuse = tasknode_exec.max_reuse_volume(tasknode_cand)
                    reused_vol_tmp[task_cand] = max_reuse
                data_reused_vol.append(reused_vol_tmp)
            task_nxt = []
            for i in range(len(data_reused_vol)):
                sorted_items = sorted(data_reused_vol[i].items(), key=lambda item: item[1],reverse=True)
                task_nxt_list = list(dict(sorted_items).keys())
                for task_tmp in task_nxt:
                    task_nxt_list.remove(task_tmp)  ## avoid repeat task
                sel_num = num_task_sel // len(data_reused_vol)
                selected_task = task_nxt_list[0:sel_num]
                task_nxt += selected_task
            remaining_task = num_task_sel - len(task_nxt)
            if remaining_task > 0:
                sorted_items = sorted(data_reused_vol[0].items(), key=lambda item: item[1], reverse=True)
                task_nxt_list = list(dict(sorted_items).keys())
                for task_tmp in task_nxt:
                    task_nxt_list.remove(task_tmp)  ## avoid repeat task
                selected_task = task_nxt_list[0:remaining_task]
                task_nxt += selected_task
            return task_nxt

    def update_ofm_valid_time(self):
        for task_name, task in self.taskDag.task_dict.items():
            children = self.taskDag.get_node_children(task_name)
            valid_order = 0
            if len(children) > 0:
                for child_name in children.keys():
                    valid_order = max(valid_order, self.taskDag.get_task_exe_time(child_name))
            else:
                valid_order = len(self.exe_order)
            self.taskDag.set_task_valid_time(task_name, valid_order)

    def map_task(self):
        '''
        map the task onto the physical cores, greedy-based data buffering strategy.
        This is a based line mapping strategy just like atomic and klotski.
        During mapping round, each core will find IFM and WEI from last round (IFM and WEI will be over writen in next round).
        If IFM and WEI are not on chip, move them according to sub-task's prevs.
        TO DO: The scheduler should make sure that all these IFM and WEI can be reuse fully (intra-layer level).
        :return:
        '''
        ## initial
        tasks_name_cur_list = self.exe_order[0]
        ubuf_manager_list = []
        task_map_tmp = []
        for i in range(self.numCore):
            ubuf_manager_list.append(bufManager(self.ubuf))
            if i < len(tasks_name_cur_list):
                task_name_cur = tasks_name_cur_list[i]
                self.taskDag.set_task_coreIdx(task_name_cur,i)  ## zigzag mapping at initial round
                ofm_volume = self.taskDag.get_ofm_volume(task_name_cur)
                ubuf_manager_list[i].store_data(task_name_cur, ofm_volume)
                task_map_tmp.append(task_name_cur)
        self.map_order.append(task_map_tmp)
        task_map_cur = task_map_tmp.copy()

        for order, task_name_nxt_list in enumerate(self.exe_order[1:]):
            ## caculate reuse data volume,
            # 1. IFM and WEI in last round can be reused,
            # 2. OFM stored in ubuf can be reused
            cur_order = order + 1
            task_inter_ifm_reuse_dict = {}
            task_intra_ifm_reuse_dict = {}
            task_wei_reuse_dict = {}
            task_ifm_reuse_dict = {}
            # task_inter_reuse_dict = {}

            task_vol_occupy = []
            for idx, task_name in enumerate(task_name_nxt_list):
                ofm_volume = self.taskDag.get_ofm_volume(task_name)
                ifm_wei_vol = self.taskDag.get_wei_volume(task_name) + self.taskDag.get_ifm_volume(task_name)
                task_vol_occupy.append(ofm_volume+ifm_wei_vol)
            task_vol_occupy_max = max(task_vol_occupy)
            task_name_max_vol = task_name_nxt_list[task_vol_occupy.index(task_vol_occupy_max)]

            ## update ubuf for next mapping, since the task can be mapped onto every core, all cores should have sufficent space:
            for idx in range(self.numCore):
                ubuf_manager = ubuf_manager_list[idx]
                free_space = ubuf_manager.free_space()
                while free_space < task_vol_occupy_max:
                    wb_name = ubuf_manager.writeback_data()
                    if self.verbose:
                        print(
                            f'The task {task_name_max_vol} will occupy {task_vol_occupy_max} bytes and make ubuf with free_space {free_space} overflow')
                        print(
                            f'Due to memory overflow, in execute order {cur_order - 1}, ofm {wb_name} in core {idx} is written back to DRAM')
                    # if wb_name == 'conv2_2_b.0.0.1':
                    #     print(cur_order)
                    free_space = ubuf_manager.free_space()
                    self.taskDag.set_task_writeback_time(wb_name, cur_order - 1)

            for i, task_name_nxt in enumerate(task_name_nxt_list):
                inter_ifm_reuse_list = []
                intra_ifm_reuse_list = []
                intra_wei_reuse_list = []
                ifm_reuse_list = []
                # reuse_total_list = []
                task_nxt = self.taskDag[task_name_nxt]
                for j, task_name_cur in enumerate(task_map_cur):
                    task_cur = self.taskDag[task_name_cur]
                    # intra-layer_reuse
                    intra_ifm_reuse = task_nxt.ifm_reuse_volume(task_cur)
                    intra_wei_reuse = task_nxt.wei_reuse_volume(task_cur)
                    intra_ifm_reuse_list += [intra_ifm_reuse]
                    intra_wei_reuse_list += [intra_wei_reuse]
                    # inter_layer_reuse
                    stored_ofm_name = ubuf_manager_list[j].get_stored_list()
                    # print(f'ubuf in core {j} has stored ofm {stored_ofm_name}')
                    inter_ifm_reuse = self.taskDag.get_ubuf_reuse(task_name_nxt, stored_ofm_name, verbose=False)
                    inter_ifm_reuse_list += [inter_ifm_reuse]
                    ifm_reuse_list += [max(intra_ifm_reuse, inter_ifm_reuse)]
                idle_core = self.numCore - len(task_map_cur)
                assert idle_core >= 0
                intra_ifm_reuse_list += [0] * idle_core
                inter_ifm_reuse_list += [0] * idle_core
                ifm_reuse_list += [0] * idle_core
                intra_wei_reuse_list += [0] * idle_core
                task_intra_ifm_reuse_dict[task_name_nxt] = intra_ifm_reuse_list
                task_inter_ifm_reuse_dict[task_name_nxt] = inter_ifm_reuse_list
                # task_intra_ifm_reuse_dict[task_name_nxt] = intra_ifm_reuse_list
                task_wei_reuse_dict[task_name_nxt] = intra_wei_reuse_list
                task_ifm_reuse_dict[task_name_nxt] = ifm_reuse_list
            # if max(task_inter_ifm_reuse_dict.values()) != 0:
            #     print(task_inter_ifm_reuse_dict)

            # best_map = self.random_mapping(task_name_nxt_list, task_inter_ifm_reuse_dict, task_intra_ifm_reuse_dict, task_wei_reuse_dict)
            best_map = self.data_aware_mapping(task_name_nxt_list, task_inter_ifm_reuse_dict, task_intra_ifm_reuse_dict, task_wei_reuse_dict)
            ## update the ubuf storage
            for i in range(self.numCore):
                task_name = best_map[i]
                ubuf_manager = ubuf_manager_list[i]
                # free_space = ubuf_manager.free_space()
                if task_name != 'NA':
                    ofm_volume = self.taskDag.get_ofm_volume(task_name)
                    ubuf_manager.store_data(task_name, ofm_volume)
                # delete the invalid ofm
                ofm_store = ubuf_manager.get_stored_list()
                for ofm_name in ofm_store:
                    valid_order = self.taskDag.get_task_valid_time(ofm_name)
                    if valid_order <= cur_order:
                        ubuf_manager.delete_data(ofm_name)
                        # print(f'In execute order {cur_order}, ofm {ofm_name} of valid order {valid_order} in core {i} is destroyed.')

            self.map_order.append(best_map)

    def data_aware_mapping(self,task_name_nxt, inter_ifm_reuse_dict, intra_ifm_reuse_dict, wei_reuse_dict):
        map_sch = ['NA' for _ in range(self.numCore)]
        task_toMap  = task_name_nxt
        xlen = self.core_mask.shape[0]
        ylen = self.core_mask.shape[1]
        max_reuse_dict = {}
        # print(f'mapping following: {task_name_nxt}')
        for task in task_name_nxt:
            if task == 'NA': continue
            inter_ifm_reuse = np.array(inter_ifm_reuse_dict[task])
            intra_ifm_reuse = np.array(intra_ifm_reuse_dict[task])
            wei_reuse = np.array(wei_reuse_dict[task])
            inter_reuse = inter_ifm_reuse + wei_reuse
            intra_reuse = intra_ifm_reuse + wei_reuse
            max_reuse = max(max(inter_reuse), max(intra_reuse))
            max_reuse_dict[task] = max_reuse
        max_reuse_sorted = sorted(max_reuse_dict.items(), key=lambda item: item[1], reverse=True)
        for task, reused_data in max_reuse_sorted:
            if reused_data == 0:
                center_cidx = ( xlen// 2,  ylen// 2)
            else:
                inter_ifm_reuse = np.array(inter_ifm_reuse_dict[task])
                intra_ifm_reuse = np.array(intra_ifm_reuse_dict[task])
                wei_reuse = np.array(wei_reuse_dict[task])
                inter_reuse = list(inter_ifm_reuse + wei_reuse)
                intra_reuse = list(intra_ifm_reuse + wei_reuse)
                idx = inter_reuse.index(reused_data) if reused_data in inter_reuse else intra_reuse.index(reused_data)
                center_cidx = self.idx2coreidx_map[idx]
            core_map_order_prior = gen_map_order((xlen, ylen), center_cidx)
            # print(task, core_map_order_prior, center_cidx)
            for cidx_cand in core_map_order_prior:
                idx_valid = []
                for cidx in cidx_cand:
                    idx = self.idx2coreidx_map.index(cidx)
                    # print(idx, cidx)
                    if map_sch[idx] == 'NA':
                        idx_valid.append(idx)
                if len(idx_valid) == 0: 
                    map_idx = None 
                    continue
                elif len(idx_valid) == 1:
                    map_idx = idx_valid[0]
                    break
                elif len(idx_valid) > 1:
                    reuse_total = []
                    for idx in idx_valid:
                        tmp = 0
                        for task_to in task_toMap:
                            inter_ifm_reuse = inter_ifm_reuse_dict[task_to]
                            intra_ifm_reuse = intra_ifm_reuse_dict[task_to]
                            wei_reuse       = wei_reuse_dict[task_to]
                            tmp += max(inter_ifm_reuse[idx], intra_ifm_reuse[idx]) + wei_reuse[idx]
                        reuse_total.append(tmp)
                    map_idx = idx_valid[reuse_total.index(min(reuse_total))]
                    break
            # if map_sch[map_idx] != 'NA':
            #     print(task_name_nxt)
            # print(map_idx, idx_valid)
            assert map_sch[map_idx] == 'NA'
            map_sch[map_idx] = task
            # if task not in task_toMap:
            #     print(task, task_toMap)
            task_toMap.remove(task)
        return map_sch


    def random_mapping(self,task_name_nxt, inter_ifm_reuse_dict, intra_ifm_reuse_dict, wei_reuse_dict):
        '''
        randomly mapping the tasks and keep the mapping scheme with minimum cost
        :param task_name_nxt: tasks name to be mapped
        :param inter_ifm_reuse_dict: for each task, a list represents inter ifm reuse time in each core
        :param intra_ifm_reuse_dict: intra ifm reuse (intra have higher priority) in
        :param wei_reuse_dict: weight data reuse in each core
        :param task_data_vol: total data volume in this task
        can't get the optimial result even compared with thermal aware mapping
        :return:
        '''
        iter_times = len(task_name_nxt) * self.numCore * 100

        # generate the mapping scheme:
        if len(task_name_nxt) < self.numCore:
            task_name_nxt += ['NA'] * (self.numCore - len(task_name_nxt))
        map_schemes = [random.sample(task_name_nxt, len(task_name_nxt)) for _ in range(iter_times)]
        ## evaluate mapping schemes
        cost = self.evaluate_mapping(task_name_nxt, inter_ifm_reuse_dict, intra_ifm_reuse_dict, wei_reuse_dict)
        best_map = task_name_nxt
        for map_scheme in map_schemes:
            cost_tmp = self.evaluate_mapping(map_scheme, inter_ifm_reuse_dict, intra_ifm_reuse_dict, wei_reuse_dict)
            if cost_tmp < cost:
                best_map = map_scheme
        return best_map

    def evaluate_mapping(self, map_scheme,inter_ifm_reuse_dict, intra_ifm_reuse_dict, wei_reuse_dict):
        cost = 0
        for idx, task_name in enumerate(map_scheme):
            if task_name != 'NA':
                ifm_wei_data_vol = self.taskDag.get_ifm_volume(task_name) + self.taskDag.get_wei_volume(task_name)
                intra_ifm_reuse = intra_ifm_reuse_dict[task_name]
                inter_ifm_reuse = inter_ifm_reuse_dict[task_name]
                wei_reuse = wei_reuse_dict[task_name]
                cost += self.evaluate_map_idx(idx,inter_ifm_reuse, intra_ifm_reuse, wei_reuse, ifm_wei_data_vol)
        return cost

    def evaluate_map_idx(self, map_idx, inter_ifm_reuse, intra_ifm_reuse, wei_reuse, ifm_wei_data_vol):
        '''
        evaluate data mapping with current buffering strategy, but we don't consider buffer overflow
        :param map_idx: mapped core index
        :param intra_ifm_reuse: intra ifm reuse list for each core
        :param inter_ifm_reuse: inter ifm reuse list for each core
        :param wei_reuse: weight data reuse list for each core
        :param data_vol_total:
        :return: cost (currently data movement )
        '''
        is_intra_ifm = max(intra_ifm_reuse) > 0
        # map_coreidx = idx2coreidx(map_idx, self.core_mask)
        map_coreidx = self.idx2coreidx_map[map_idx]
        if is_intra_ifm:
            dram_move_vol = ifm_wei_data_vol - max(intra_ifm_reuse) - max(wei_reuse)
            max_ifm_reuse = max(intra_ifm_reuse)
            max_wei_reuse = max(wei_reuse)
            ifm_src_idx = intra_ifm_reuse.index(max_ifm_reuse)
            wei_src_idx = wei_reuse.index(max_wei_reuse)
            # ifm_src_coreidx = idx2coreidx(ifm_src_idx, self.core_mask)
            # wei_src_coreidx = idx2coreidx(wei_src_idx, self.core_mask)
            ifm_src_coreidx = self.idx2coreidx_map[ifm_src_idx]
            wei_src_coreidx = self.idx2coreidx_map[wei_src_idx]
            ## TO DO: considering DRAM access from different idx/channel to explore the thermal problem of DRAM
            # dram_dist = min(abs(self.xlen - map_coreidx[0]), map_coreidx[0])
            dram_dist = 0
            cost = (dram_move_vol * dram_dist + max_ifm_reuse * manhattan_distance(ifm_src_coreidx, map_coreidx)
                        + max_wei_reuse * manhattan_distance(wei_src_coreidx, map_coreidx))
        else:
            dram_data_move = ifm_wei_data_vol - sum(inter_ifm_reuse) - max(wei_reuse)
            max_wei_reuse = max(wei_reuse)
            distances = np.zeros(self.numCore)
            for i in range(self.numCore):
                # coreidx = idx2coreidx(i, self.core_mask)
                coreidx = self.idx2coreidx_map[i]
                distances[i] = manhattan_distance(coreidx, map_coreidx)
            wei_src_idx = wei_reuse.index(max_wei_reuse)
            # wei_src_coreidx = idx2coreidx(wei_src_idx, self.core_mask)
            wei_src_coreidx = self.idx2coreidx_map[wei_src_idx]
            ## TO DO: considering DRAM access from different idx/channel to explore the thermal problem of DRAM
            # dram_dist = min(abs(self.xlen - map_coreidx[0]), map_coreidx[0])
            dram_dist = 0
            wei_dist = manhattan_distance(wei_src_coreidx, map_coreidx)
            cost = dram_data_move * dram_dist + max_wei_reuse * wei_dist + np.dot(distances,inter_ifm_reuse)
        return cost

    def thermal_aware_task_map(self):
        ## initial the thermal effect factor and ubuf manager
        lateral_factor = dict()    ## the lateral affect factor for all the cores
        ubuf_manager_list = list()
        for i in range(self.numCore):
            ubuf_manager_list.append(bufManager(self.ubuf))
            factor = 0    
            src_coreidx = self.idx2coreidx_map[i]
            for j in range(self.numCore):
                dst_coreidx = self.idx2coreidx_map[j]
                factor += eu_distance_sqrt(src_coreidx, dst_coreidx)
            lateral_factor[src_coreidx] = factor
        lateral_factor = dict(sorted(lateral_factor.items(), key=lambda item: item[1], reverse=True))
        core_order = [key for key in lateral_factor.keys()]
        ## caculate the eneregy cost and map the highest-cost task to the core with the best cooling condition
        for cur_order, task_names in enumerate(self.exe_order):
            energy_cost = dict()
            map_sch = ['NA' for _ in range(self.numCore)]
            for name in task_names:
                l0a_read_size = l0a_write_size = l0b_read_size = l0b_write_size = l0c_read_size = l0c_write_size = 0
                l1c_read_size = l1c_write_size = ubuf_read_size = ubuf_write_size = 0
                tot_energy  = 0
                task_node =  self.taskDag[name]
                ofm_volume = task_node.toVolume()
                ifm_volume = task_node.fromVolume()
                wei_volume = task_node.total_filter_size()
                bo, ho, wo, co = task_node.get_ofm_shape()
                ci = task_node.fromrange.cRange.length()
                if isinstance(task_node, CONVTaskNode) or isinstance(task_node, GemmTaskNode):
                    hfil, wfil = task_node.hfil, task_node.wfil
                    mac_tot_ops = task_node.total_ops()
                    padding_rate = task_node.get_padding_rate()
                    ci = 1 if isinstance(task_node, DPConvTaskNode) else ci
                    (hw_utl, co_ult, cin_utl) = self.cluster.cal_mtxu_utl([ho,wo,ci,co])
                    l0a_read_size += mac_tot_ops * (1 - padding_rate) /  self.cluster.w_pe
                    l0b_read_size += mac_tot_ops  / self.cluster.h_pe
                    ifm_vol_perMM = int(self.cluster.h_pe * ci * hfil * wfil / task_node.ifm_hw_overlap())
                    wei_vol_perMM = int(self.cluster.w_pe * ci * hfil * wfil)
                    psum_max_perRF = self.cluster.get_buf_size('l0c') // self.cluster.w_pe // self.cluster.get_buf_bw('l0c')
                    l0a_fetch_time_perMM = ifm_vol_perMM / self.cluster.get_buf_size('l0a')
                    l0b_fetch_time_perMM = wei_vol_perMM / self.cluster.get_buf_size('l0b')
                    l0a_fetch_time_hw = ho * wo / self.cluster.h_pe
                    l0b_fetch_time_co = co / self.cluster.w_pe
                    l0a_fetch_time_tot = l0a_fetch_time_perMM * l0a_fetch_time_hw
                    l0b_fetch_time_tot = l0b_fetch_time_perMM * l0b_fetch_time_co
                    if l0a_fetch_time_tot <=1 or l0b_fetch_time_tot <= 1:
                        ## l0a or l0b can store all ifm/wei at one time
                        l0a_write_size += ifm_volume
                        l0b_write_size += wei_volume
                        l0c_write_size += ofm_volume
                    elif l0a_fetch_time_perMM <= 1 or l0b_fetch_time_perMM <= 1:
                        ## l0a or l0b can store >1 data volum of perMM
                        aloop = math.ceil(ifm_volume / self.cluster.get_buf_size('l0a'))
                        bloop = math.ceil(wei_volume / self.cluster.get_buf_size('l0b'))
                        l0c_write_size += ofm_volume
                        if ifm_volume * bloop  > wei_volume * aloop:  # fix ifm and move wei
                            l0a_write_size += ifm_volume
                            l0b_write_size += wei_volume * bloop
                        else:
                            l0a_write_size += ifm_volume * aloop
                            l0b_write_size += wei_volume
                    else:
                        ## l0a or l0b can't store the data of per MM, 
                        ## can't fix WEI/IFM in l0 buffer, hence l0a, l0b need read data from ubuf multiple times 
                        l0a_fetch_time_perMM = math.ceil(l0a_fetch_time_perMM)
                        l0b_fetch_time_perMM = math.ceil(l0b_fetch_time_perMM)
                        l0a_loop_times = math.ceil(l0b_fetch_time_co)
                        l0b_loop_times = math.ceil(l0a_fetch_time_hw)
                        ## considering the l0c can store the partial sum, hence the ifm/wei can be fixed psum_max_perRF times
                        if l0a_loop_times * ifm_volume > l0b_loop_times * wei_volume:
                            l0c_psum_reuse = min(l0a_loop_times, psum_max_perRF)
                            l0a_loop_times = math.ceil(l0a_loop_times / l0c_psum_reuse)
                        else:
                            l0c_psum_reuse = min(l0b_loop_times, psum_max_perRF)
                            l0b_loop_times = math.ceil(l0b_loop_times / l0c_psum_reuse)
                        l0a_write_size += ifm_volume * l0a_loop_times
                        l0b_write_size += wei_volume * l0b_loop_times
                        l0c_write_size += ofm_volume * l0c_psum_reuse  
                    ubuf_read_size += l0a_write_size + l0b_write_size
                    l0c_write_size += mac_tot_ops / self.cluster.h_pe
                    l0c_read_size += ofm_volume
                    l1c_read_size += ofm_volume
                    l1c_write_size += ofm_volume
                    ubuf_write_size += ofm_volume
                    comp_energy = mac_tot_ops * self.cluster.mtxu_cost * (1 - padding_rate)
                    
                elif isinstance(task_node, LRLTaskNode):
                    ifm_volume = task_node.fromVolume()
                    ofm_volume = task_node.toVolume()
                    vecu_tot_ops = task_node.total_ops()
                    ubuf_read_size += ifm_volume
                    ubuf_write_size += ofm_volume
                    comp_energy = vecu_tot_ops * self.cluster.vecu_cost
                else:
                    raise TypeError('Task node type not supported')

                l0a_cost = self.cluster.get_buf_rcost('l0a') * l0a_read_size + self.cluster.get_buf_wcost('l0a') * l0a_write_size
                l0b_cost = self.cluster.get_buf_rcost('l0b') * l0b_read_size + self.cluster.get_buf_wcost('l0b') * l0b_write_size 
                l0c_cost = self.cluster.get_buf_rcost('l0c') * l0c_read_size + self.cluster.get_buf_wcost('l0c') * l0c_write_size 
                l1c_cost = self.cluster.get_buf_rcost('l1c') * l1c_read_size + self.cluster.get_buf_wcost('l1c') * l1c_write_size 
                ubuf_cost= self.cluster.get_buf_rcost('ubuf') * ubuf_read_size + self.cluster.get_buf_wcost('ubuf') * ubuf_write_size 
                tot_energy = comp_energy + l0a_cost + l0b_cost + l0c_cost + l1c_cost + ubuf_cost
                energy_cost[name] = tot_energy
            energy_cost = dict(sorted(energy_cost.items(), key=lambda item: item[1], reverse=True))
            for i, task_name in enumerate(energy_cost.keys()):
                map_cidx = core_order[i]
                map_idx = self.idx2coreidx_map.index(map_cidx)
                assert map_sch[map_idx] == 'NA' 
                map_sch[map_idx] = task_name
            # manage the ubuf 
            if self.baseline1:
                for i, task_name in enumerate(map_sch):
                    if task_name == 'NA': continue
                    self.taskDag.set_task_writeback_time(task_name, cur_order)
            else:
                for i, task_name in enumerate(map_sch):
                    ubuf_manager = ubuf_manager_list[i]
                    # delete the invalid ofm
                    ofm_store = ubuf_manager.get_stored_list()
                    for ofm_name in ofm_store:
                        valid_order = self.taskDag.get_task_valid_time(ofm_name)
                        if valid_order <= cur_order:
                            ubuf_manager.delete_data(ofm_name)
                            # print(f'In execute order {cur_order}, ofm {ofm_name} of valid order {valid_order} in core {i} is destroyed.')
                    if task_name == 'NA': continue
                    task_node = self.taskDag[task_name]
                    ofm_volume = task_node.toVolume()
                    ifm_volume = task_node.fromVolume()
                    wei_volume = task_node.total_filter_size()
                    occupy_vol = ifm_volume + ofm_volume + wei_volume
                    # print(f'{task_name} : will use {occupy_vol}')
                    free_space = ubuf_manager.free_space()
                    ## write back ofm according to its valid time
                    ofm_vtime = {}
                    if free_space < occupy_vol:
                        ofm_store = ubuf_manager.get_stored_list()
                        for ofm_name in ofm_store:
                            ofm_vtime[ofm_name] = self.taskDag.get_task_valid_time(ofm_name)
                    sorted_ofm_vtime = sorted(ofm_vtime.items(), key=lambda item: item[1], reverse=True)
                    wb_order = list(OrderedDict(sorted_ofm_vtime).keys())
                    while free_space < occupy_vol:
                        assert cur_order > 0
                        wb_name = wb_order[0]
                        ubuf_manager.delete_data(wb_name)
                        free_space = ubuf_manager.free_space()
                        self.taskDag.set_task_writeback_time(wb_name, cur_order - 1)
                        wb_order.pop(0)
                    ## greedy writeback, always write back the first-in ofm
                    # while free_space < occupy_vol:
                    #     assert cur_order > 0
                    #     wb_name = ubuf_manager.writeback_data()
                    #     if self.verbose:
                    #         print(f'The task {task_name} will occupy {occupy_vol} bytes and make ubuf with free_space {free_space} overflow')
                    #         print(f'Due to memory overflow, in execute order {cur_order - 1}, ofm {wb_name} in core {i} is written back to DRAM')
                    #     free_space = ubuf_manager.free_space()
                    #     self.taskDag.set_task_writeback_time(wb_name, cur_order - 1)
                    ubuf_manager.store_data(task_name, ofm_volume)
            self.map_order.append(map_sch)
    
    def dump_execute_order(self):
        str_ = ''
        if len(self.exe_order) == 0:
            str_ = 'The scheduler do not run'
        else:
            str_ += 'Execution Order is:\n'
            for i, task_exe in enumerate(self.exe_order):
                str_ += '\t' + str(i) + ': ' + str(task_exe) + '\n'
        return str_

    def clear(self):
        self.taskDag.clear()
        self.exe_order.clear()
        self.map_order.clear()

    def __str__(self):
        str_ = ''
        if len(self.exe_order) == 0:
            str_ = 'The scheduler do not run'
        else:
            # str_ += 'Execution Order is:\n'
            # for i, task_exe in enumerate(self.exe_order):
            #     str_ += '\t' + str(i) + ': ' + str(task_exe) + '\n'
            str_ += 'Mapping Order is:\n'
            for i, task_map in enumerate(self.map_order):
                str_ += '\t' + str(i) + ': ' + str(task_map) + '\n'
        return str_