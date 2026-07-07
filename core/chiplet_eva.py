import numpy as np

from nns import import_network
from core.accCore import AccCore, Onchip_buffer, Core_mem_hierarchy, Cluster
from core.partengine import LayerTree
from core.taskdag import TaskDAG
from core.schedule import Schedule
from core.nop import Nop
from core.evaluator import Evaluator
from core.statistic import Statistic
from core.gen_floorplan import floorplan_generator, floorplan_generator_3D
from core.gen_hw_setting import *
from core.tech_params import TechParams
from core.util import find_hotpoint

class chiplet_evaluator:

    def __init__(self , hotspot_path, sim_path ,sys_info:list, thermal_map = True , baseline1=False, baseline2=False, baseline3= False,wkld_idpdt = False, clock_freq = 1.8e9, tech: TechParams = None, workload_subset: list = None):
        # baseline 1: chiplet-gym, not data buffering
        # baseline 2: TESA, not NoP/NoC
        # wkld_idpdt: peak temp. is max(peak_temp)
        # tech: technology assumptions (D2D interconnect energy, yield/defect-density model).
        # Defaults to TechParams(), which reproduces ThermoDSE's original fixed-technology behavior.
        self.tech = tech if tech is not None else TechParams()
        self.baseline1 = baseline1
        self.baseline2 = baseline2
        self.baseline3 = baseline3
        self.thermal_map = thermal_map
        self.wkld_idpdt = wkld_idpdt
        all_nets = ['resnet50', 'googlenet', 'unet', 'mobilenet', 'yolo', 'transformer']
        all_b_tot = [2, 2, 2, 4, 4, 1]
        all_b_exe = [2, 2, 1, 2, 4, 1]
        all_sparsty = [0.217, 0.375, 0.453, 0.368, 0.009, 0]
        # workload_subset: evaluate only these networks instead of all 6 -- a cheap-fidelity
        # proxy (see docs/algorithm.md's BiSTCO-TRACE). Must be validated (rank correlation /
        # top-K recall against the full 6-workload score) before trusting it to screen candidates;
        # defaults to None, which reproduces the original full-6-workload behavior exactly.
        if workload_subset is None:
            self.nets, self.b_tot, self.b_exe, self.sparsty = all_nets, all_b_tot, all_b_exe, all_sparsty
        else:
            unknown = set(workload_subset) - set(all_nets)
            if unknown:
                raise ValueError(f'workload_subset contains unknown network(s): {unknown}')
            indices = [all_nets.index(name) for name in workload_subset]
            self.nets = [all_nets[i] for i in indices]
            self.b_tot = [all_b_tot[i] for i in indices]
            self.b_exe = [all_b_exe[i] for i in indices]
            self.sparsty = [all_sparsty[i] for i in indices]
        self.clock_freq = clock_freq
        # self.interposer_cons = interposer_cons
        self.hotspot_path = hotspot_path
        self.sim_path = sim_path
        self.sys_info = sys_info
        self.chipletX, self.chipletY = sys_info[0], sys_info[1]
        self.chipletCx, self.chipletCy  = sys_info[2], sys_info[3]
        self.chiplet_intvl = sys_info[4]
        self.mtxu_h, self.mtxu_w = sys_info[5], sys_info[6]
        self.ubuf_size = sys_info[7]
        self.nop_bw = sys_info[8]
        # self.batch = sys_info[7]
        # self.closed_core = [sys_info[9], sys_info[10], sys_info[11]]
        self.dram_bw_design = sys_info[9]
        # print(self.closed_core)
        ## the rest arch parameter can be got from above parameters
        self.vecu = self.mtxu_w
        self.ubuf_bw = self.mtxu_w + self.mtxu_h
        self.l0a_size = self.ubuf_size // 8
        self.l0b_size = self.ubuf_size // 8 
        self.l1c_size = self.ubuf_size // 8
        self.l0c_size = 64 * self.mtxu_w  * 3  # Fix each RF deepth is 16, total mtxu_w RFs, fix 24bit for partital sum
        self.l0a_bw = self.mtxu_h
        self.l0b_bw = self.mtxu_w
        self.l0c_bw = 3
        self.l1c_bw = self.mtxu_w
        self.noc_bw = sys_info[8]
        # nop_link area
        if baseline2: self.nop_bw = self.dram_bw_design // (self.chipletX * self.chipletY) 
        nop_area_link, nop_bit_cost = nop_setting_gen(self.chiplet_intvl * 1000, self.nop_bw, tech=self.tech)
        if self.chipletCx == 1 or self.baseline2:
            nop_area_x = nop_area_link * 0 * 1e-12 ## um^2 -> mm^2
        elif self.chipletCx == 2:
            nop_area_x = nop_area_link * (self.chipletX / self.chipletCx) * 1e-12 ## um^2 -> mm^2
        else:
            nop_area_x = nop_area_link * (self.chipletX / self.chipletCx) * 2 * 1e-12 ## um^2 -> mm^2
        
        if self.chipletCy == 1 or self.baseline2:
            nop_area_y = nop_area_link * 0 * 1e-12 ## um^2 -> mm^2
        elif self.chipletCy == 2:
            nop_area_y = nop_area_link * (self.chipletY / self.chipletCy) * 1e-12 ## um^2 -> mm^2
        else:
            nop_area_y = nop_area_link * (self.chipletY / self.chipletCy) * 2 * 1e-12 ## um^2 -> mm^2
        self.nop_area = nop_area_x +nop_area_y
        #DRAM and IO_DIE area:
        DDR_PHY_den = 6.53 * 1e6
        DDR_ctrl_den = 510 * 1000 * 0.09
        if self.baseline2 or self.baseline3:   ## for baseline2, since there is not any noc/nop, we only evaluate 1 core
            self.dram_list = [(0, self.chipletY -1)]
            self.dram_bw = self.dram_bw_design // (self.chipletX * self.chipletY) 
            # print(self.dram_bw)
            PCIe_area = 300000 * self.dram_bw / 128  ## um^2 
            self.IO_die_area_each = (self.dram_bw / 44 * (DDR_ctrl_den + DDR_PHY_den) + PCIe_area + PCIe_area + nop_area_link * 2) * 1e-12 #um^2 -> mm^2
        else:
            if self.chipletY == 1:
                self.dram_list = [(0,0)]
            else:
                self.dram_list = [(0,0), (0, self.chipletY -1), (self.chipletX + 1, 0), (self.chipletX + 1, self.chipletY - 1)]
            self.dram_bw = self.dram_bw_design // len(self.dram_list) 
            PCIe_area = 300000 * self.dram_bw / 128  ## um^2 
            self.IO_die_area_each = (self.dram_bw / 44 * (DDR_ctrl_den + DDR_PHY_den) + PCIe_area + nop_area_link * 2 ) * 1e-12 # um^2 -> mm^2

        self.nop_cost = nop_bit_cost * 8
        self.noc_cost = 0.6 * 8
        self.dram_cost = 8 * 8

        ## objects
        self.energy = 0
        self.latency = 0
        self.die_yield = 0
        self.peak_temp = 0

    def generate_hardware(self):
        # l0a, l0b, l0c are ping-pong buffer, thus the area should be mutiplied 2
        # pr_density is the density of placement and routing, logic_density is the logic gate density for each module
        pr_density = 0.5
        area_factor =  4.375  * pr_density
        energy_factor = 1.947
        smtxu = (self.mtxu_h, self.mtxu_w)
        lib_type = self.tech.lib_type
        ubuf_area, ubuf_rcost, ubuf_wcost = sram_setting_gen(self.ubuf_size,area_factor,energy_factor,lib_type=lib_type)
        l0a_area, l0a_rcost, l0a_wcost = sram_setting_gen(self.l0a_size,area_factor / 2,energy_factor,lib_type=lib_type)
        l0b_area, l0b_rcost, l0b_wcost = sram_setting_gen(self.l0b_size,area_factor / 2,energy_factor,lib_type=lib_type)
        l0c_area, l0c_rcost, l0c_wcost = regf_setting_gen(self.l0c_size,area_factor,energy_factor,lib_type=lib_type)
        l1c_area, l1c_rcost, l1c_wcost = sram_setting_gen(self.l1c_size,area_factor / 2,energy_factor,lib_type=lib_type)
        mtxu_area, mtxu_cost = mtxu_settting_gen(smtxu, area_factor, energy_factor,lib_type=lib_type)
        vecu_area, vecu_cost = vecu_settting_gen(self.vecu, area_factor ,energy_factor,lib_type=lib_type)
        ubuf = Onchip_buffer('ubuf', self.ubuf_size, ubuf_rcost, ubuf_wcost, ubuf_area, self.ubuf_bw)
        l0a = Onchip_buffer('l0a', self.l0a_size, l0a_rcost, l0a_wcost, l0a_area, self.l0a_bw)
        l0b = Onchip_buffer('l0b', self.l0b_size, l0b_rcost, l0b_wcost, l0b_area , self.l0b_bw)
        l0c = Onchip_buffer('l0c', self.l0c_size, l0c_rcost *self.l0c_bw, l0c_wcost * self.l0c_bw, l0c_area, self.l0c_bw)
        l1c = Onchip_buffer('l1c', self.l1c_size, l1c_rcost, l1c_wcost, l1c_area, self.l1c_bw)
        acc_mem = Core_mem_hierarchy()
        acc_mem.add_memory(ubuf)
        acc_mem.add_memory(l0a)
        acc_mem.add_memory(l0b)
        acc_mem.add_memory(l0c)
        acc_mem.add_memory(l1c)
        self.mtxu_cost_full = mtxu_cost   ## for different NNs, the mtxu_cost should be multiplied the density of input data
        self.core_spec = {}
        self.core_spec['mtxu'] = [mtxu_area, mtxu_cost]
        self.core_spec['vecu'] = [vecu_area, vecu_cost]
        self.core_spec['ubuf'] = [ubuf_area, ubuf_rcost, ubuf_wcost]
        self.core_spec['l0a'] = [l0a_area, l0a_rcost, l0a_wcost]
        self.core_spec['l0b'] = [l0b_area, l0b_rcost, l0b_wcost]
        self.core_spec['l0c'] = [l0c_area, l0c_rcost, l0c_wcost]
        self.core_spec['l1c'] = [l1c_area, l1c_rcost, l1c_wcost]
        for name, info in self.core_spec.items():
            print(f'{name}:{info[0] * self.chipletX*self.chipletY / 0.0002 * 100}%')

        self.cluster = Cluster((self.chipletX, self.chipletY),smtxu, self.vecu,mtxu_cost, vecu_cost, acc_mem)
        self.flp_generator = floorplan_generator_3D(self.core_spec, path=self.sim_path, hs_path = self.hotspot_path)
        self.flp_generator.gen_core_floorplan()
        self.die_h_list, self.die_w_list, self.sys_h, self.sys_w = self.flp_generator.gen_sys_floorplan_nonunf(self.chiplet_intvl, self.chipletX, self.chipletY, self.chipletCx, self.chipletCy, NoC_space=0.0005)
        self.monitor = Statistic((self.chipletX, self.chipletY), (self.chipletCx, self.chipletCy), smtxu, smtxu, self.clock_freq)
        
    def evaluate(self, draw_fig=False):
        if self.baseline2 or self.baseline3:
            core_mask = np.full((self.chipletX, self.chipletY), False)
            core_mask[0][0] = True
        else:
            core_mask = np.full((self.chipletX, self.chipletY), True)
        IO_die_num = (self.chipletX * self.chipletY) if self.baseline2 or self.baseline3 else len(self.dram_list)
        chiplet_area = self.sys_h * self.sys_w + self.IO_die_area_each * IO_die_num
        # print(f'({self.sys_h}, {self.sys_w}), total area: {chiplet_area}')
        # if self.sys_h > self.interposer_cons or self.sys_h > self.interposer_cons:
        #     return 1, 1, 1e7
        y_tmp = 0
        latency_nn_max = 0
        for die_h in self.die_h_list:
            for die_w in self.die_w_list:
                y_tmp = yield_setting_gen(die_h * die_w + self.nop_area, tech=self.tech)
                self.die_yield += y_tmp * (die_h * die_w / (sum(self.die_h_list) * sum(self.die_w_list)))
                # print(f'compute area:{die_h * die_w}, yeild:{y_tmp}')
        for i, net in enumerate(self.nets):
            if 'unet' in net:
                network = import_network('unet')
            else:
                network = import_network(net)
            b_exe =  1 if self.baseline2 or self.baseline3 else self.b_exe[i]
            batch_factor = 1 if self.wkld_idpdt else self.b_tot[i] // b_exe
            closed_core = 0
            mtxu_ops_cost = self.mtxu_cost_full * (1- self.sparsty[i])
            self.cluster.set_mtxu_ops_cost(mtxu_ops_cost)
            network.traverese_layer(check=False)
            ltree = LayerTree(network, self.cluster)
            ltree.init_part(tot_cores= np.sum(core_mask) - closed_core, batch=b_exe)
            ltree.gen_part_sch()
            # print(ltree)
            taskdag = TaskDAG(network.net_name)
            taskdag.generate_taskdag(network, ltree.part_sch_dict, b_exe)
            dat_depth = taskdag.traverse_taskdag()
            # taskdag.check_thermal_char()
            scheduler = Schedule(core_mask, taskdag, self.cluster, baseline1=self.baseline1 ,verbose=False)
            scheduler.schedule_dag(closed_core)
            scheduler.update_ofm_valid_time()
            if self.thermal_map == False:
                scheduler.map_task()
            else:
                scheduler.thermal_aware_task_map()
            # print(scheduler)
            nop = Nop(self.nop_bw, self.noc_bw, self.dram_bw, self.dram_list, self.nop_cost, self.noc_cost, self.dram_cost, (self.chipletX, self.chipletY), self.chipletCx, self.chipletCy)
            net_name = self.monitor.init_exe_info(network.net_name, len(scheduler.map_order))
            eva = Evaluator(net_name, taskdag, scheduler.map_order, nop, self.cluster, core_mask, self.monitor)
            eva.evaluate()
            latency_tmp, energy_tmp = self.monitor.get_nn_cost(net_name)
            latency_tmp = (latency_tmp * batch_factor)
            energy_tmp = (energy_tmp * batch_factor)
            self.latency += latency_tmp
            self.energy += energy_tmp
            if latency_tmp > latency_nn_max:
                latency_nn_max = latency_tmp
            # print(f'{net_name}-> latency:{latency_tmp}, energy:{energy_tmp}, current_max:{latency_nn_max}')
            self.monitor.cost_times(network.net_name, batch_factor)
            if self.wkld_idpdt:
                self.monitor.gen_all_ptrace_3D(isRunBaseline3=self.baseline3, gen_path=self.sim_path +'/ptrace')
                self.flp_generator.run_hotspot(draw_fig=draw_fig)
                peak_temp = find_hotpoint(file= self.sim_path + '/outputs/gcc.grid.steady')
                if peak_temp > self.peak_temp:
                    self.peak_temp = peak_temp
                # print(f'{net_name}-> peak temperature:{peak_temp}, latency:{latency_tmp}, energy:{energy_tmp}')
                ltree.clear()
                taskdag.clear()
                scheduler.clear()
                eva.clear()
                self.monitor.clear()
                self.cluster.clear()
        if self.baseline3:
            self.latency = latency_nn_max * len(self.nets)  / 1e6 # ns -> ms
        else:
            self.latency = self.latency / 1e6  # ns -> ms
        if self.baseline2 or self.baseline3: self.latency = self.latency/(self.chipletX * self.chipletY)
        self.energy = self.energy / 1e9 # pJ -> mJ
        # print(f'total delay:{self.latency / self.clock_freq * 1e9 } ms')
        # print(f'For {net_name}, E: {energy} D: {latency}, Yield: {die_yield}. The EDYP cost is {latency * energy / die_yield}')
        self.monitor.report_network_stats()
        if self.baseline2 or self.baseline3 : self.monitor.cost_copy()
        if self.wkld_idpdt == False:
            self.monitor.gen_all_ptrace_3D( isRunBaseline3=self.baseline3,  gen_path=self.sim_path + '/ptrace')
            self.flp_generator.run_hotspot(draw_fig=draw_fig)
            self.peak_temp = find_hotpoint(file= self.sim_path + '/outputs/gcc.grid.steady')
        print(f'sys_info:{self.sys_info}, \n area: {chiplet_area}, peak temperature is {self.peak_temp} K, Yield: {self.die_yield}, EDYP:{self.energy * self.latency / self.die_yield}')
        ltree.clear()
        taskdag.clear()
        scheduler.clear()
        eva.clear()
        self.monitor.clear()
        self.cluster.clear()
        return self.latency, self.energy, self.die_yield

    def get_IO_die_area(self):
        return self.IO_die_area_each * len(self.dram_list) if self.baseline2 == False and self.baseline3 == False else self.IO_die_area_each * (self.chipletX * self.chipletY)

    def get_compute_die_area(self):
        return self.sys_h * self.sys_w

    def evaluate_edyp(self):
        return self.latency, self.energy, self.die_yield

    def evaluate_area(self):
        return self.sys_h * self.sys_w + self.get_IO_die_area()
    
    
    def evaluate_thermal(self):
        return self.peak_temp



    
