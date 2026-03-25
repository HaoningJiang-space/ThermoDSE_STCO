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

import argparse
import sys

sys.path.append('../')

from core.layer import *
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
from core.util import find_hotpoint


KILO = 1024.
MILLION = 1024.*1024.

STR_FMT_NAME_LEN = '30'
STR_FMT_NUMB_LEN = '12'
STR_FMT_NUMB_PCS = '2'

STR_FMT_NAME = '{:' + STR_FMT_NAME_LEN + 's}'
STR_FMT_NUMB_HDER = '{:>' + STR_FMT_NUMB_LEN + '}'
STR_FMT_NUMB = '{:' + STR_FMT_NUMB_LEN + '.' + STR_FMT_NUMB_PCS + 'f}'

def layer_stats(args):
    # Hardware initalization assuming the clock frequency is 1GHz
    AREA_SCALE_10 = 11.667
    E_SCALE_10 = 2.467 
    AREA_SCALE_7 = 31.818
    E_SCALE_7 = 3.364
    clock = 1e9
    chip_ics = 0.0025
    acc_xlen = 6
    acc_ylen = 6
    xcut = 6
    ycut = 6
    smtxu = (32, 64)
    smtxu_grid = (16,16)
    svecu = smtxu[0]
    ubuf_cap = 0.5 *1024 *KILO
    l0a_cap = l0b_cap = l1c_cap = 64 *KILO 
    l0c_cap = 16*KILO 
    noc_bw = 128
    nop_bw = 64
    # dram_bw = int(acc_xlen * acc_ylen * smtxu[0] * smtxu[1] *2 * clock *2 / 1e12)  # 1GB/s per TOPS (AL. 0.5, 1)
    nop_area, nop_bit_cost = nop_setting_gen(chip_ics * 1000, nop_bw)
    nop_cost = nop_bit_cost * 8   ## simba 0.82-1.75 pJ/bit
    noc_cost = 0.31 * 8
    dram_acc_cost = 7.5*8   ## SET/Aotmic HBM
    logic_density = 0.6
    dram_list = [(0,0), (0, acc_ylen -1), (acc_xlen + 1, 0), (acc_xlen + 1, acc_ylen - 1)]
    # for i in range(acc_ylen):
    #     dram_list.append((0, i))
    #     dram_list.append((acc_xlen + 1, i))
    dram_bw = 128 // len(dram_list)
    ubuf_area, ubuf_rcost, ubuf_wcost = sram_setting_gen(ubuf_cap,logic_density,1)
    l0a_area, l0a_rcost, l0a_wcost = sram_setting_gen(l0a_cap,logic_density,1)
    l0b_area, l0b_rcost, l0b_wcost = sram_setting_gen(l0b_cap,logic_density,1)
    l0c_area, l0c_rcost, l0c_wcost = regf_setting_gen(l0c_cap,logic_density,1)
    l1c_area, l1c_rcost, l1c_wcost = sram_setting_gen(l1c_cap,logic_density,1)
    mtxu_area, mtxu_cost = mtxu_settting_gen(smtxu, logic_density,1)
    vecu_area, vecu_cost = vecu_settting_gen(svecu, logic_density,1)
    ubuf = Onchip_buffer('ubuf', ubuf_cap, ubuf_rcost, ubuf_wcost,1, smtxu[0]*2)
    l0a = Onchip_buffer('l0a', l0a_cap, l0a_rcost, l0a_wcost,1, smtxu[0])
    l0b = Onchip_buffer('l0b', l0b_cap, l0b_rcost, l0b_wcost,1,  smtxu[1])
    l0c = Onchip_buffer('l0c', l0c_cap, l0c_rcost, l0c_wcost,1,  smtxu[1])
    l1c = Onchip_buffer('l1c', l1c_cap, l1c_rcost, l1c_wcost,1, smtxu[0])
    acc_mem = Core_mem_hierarchy()
    acc_mem.add_memory(ubuf)
    acc_mem.add_memory(l0a)
    acc_mem.add_memory(l0b)
    acc_mem.add_memory(l0c)
    acc_mem.add_memory(l1c)
    spec_info = dict()
    spec_info['mtxu'] = [mtxu_area, mtxu_cost]
    spec_info['vecu'] = [vecu_area, vecu_cost * 0.1]
    spec_info['ubuf'] = [ubuf_area, ubuf_rcost * 0.1]
    spec_info['l0a']  = [l0a_area, l0a_rcost * 0.1]
    spec_info['l0b']  = [l0b_area, l0b_rcost * 0.1]
    spec_info['l0c']  = [l0c_area, l0c_rcost * 0.5]
    spec_info['l1c']  = [l1c_area, l1c_rcost * 0.1]
    connectivity = ['l0a\tmtxu\t1\n','l0b\tmtxu\t1\n','mtxu\tvecu\t1\n','vecu\tl0c\t1\n',
                    'ubuf\tl0a\t1\n','ubuf\tl0b\t1\n','l0c\tl1c\t1\n' , 'l1c\tubuf\t1\n']
    # acc_core = AccCore((32,32), 32, 1, 1,  acc_mem)
    comp_cluster = Cluster((acc_xlen, acc_ylen), smtxu, 32, mtxu_cost, vecu_cost, acc_mem)
    
    ''' Print stats of layers in the network. '''
    flp_generator_3d = floorplan_generator_3D(spec_info)
    flp_generator_3d.gen_core_floorplan()
    die_w, die_h, chiplet_h, chiplet_w = flp_generator_3d.gen_sys_floorplan(w_const=0.030, h_const=0.030, ics=chip_ics,xx=acc_xlen,yy=acc_ylen, xcut=xcut, ycut=ycut)
    y_die = yield_setting_gen(die_h * die_w)
    
    wkld_core_mask = np.full((4,acc_xlen,acc_ylen), False)
    wkld_core_mask[0,:,:] = True
    # wkld_core_mask[0,2,2] = False
    wkld_core_mask[1,0:3,3: ] = True
    # wkld_core_mask[1,2,3] = False
    wkld_core_mask[2,3: ,0:3] = True
    # wkld_core_mask[2,3,2] = False
    wkld_core_mask[3,3: ,3: ] = True
    # wkld_core_mask[3,3,3] = False
    monitor = Statistic((acc_xlen, acc_ylen),(xcut, ycut), smtxu, smtxu_grid, clock)
    for i, net in enumerate(args.net):
        network = import_network(net)
        core_mask = wkld_core_mask[i,:,:]
        dram_bw_net = dram_bw 
        ltree = LayerTree(network, comp_cluster)
        ltree.init_part()
        ltree.gen_part_sch()
        # print(ltree)
        taskdag = TaskDAG(network.net_name)
        taskdag.generate_taskdag(network, ltree.part_sch_dict, batch=2)
        deepth = taskdag.traverse_taskdag()
        # taskdag.check_thermal_char()
        scheduler = Schedule(core_mask, taskdag, comp_cluster, verbose=True)
        scheduler.schedule_dag()
        scheduler.update_ofm_valid_time()
        scheduler.thermal_aware_task_map()
        # print(scheduler)
        nop = Nop(nop_bw, noc_bw, dram_bw_net, dram_list, nop_cost,noc_cost, dram_acc_cost, (acc_xlen, acc_ylen), xcut, ycut)
        new_nn_name = monitor.init_exe_info(network.net_name, len(scheduler.map_order))
        eva = Evaluator(new_nn_name, taskdag, scheduler.map_order, nop, comp_cluster,core_mask, monitor)
        eva.evaluate()
        print(eva)
        latency, energy = monitor.get_nn_cost(network.net_name)
        print(f'For {net}, E: {energy} D: {latency}, Yield: {y_die}. The EDYP cost is {latency * energy / y_die}')
    ## for thermal simulation, prepare related files and call hotspot
    # monitor.gen_ptrace(network.net_name)
    # monitor.gen_ptrace_3D(network.net_name,chip_ics)
    monitor.gen_all_ptrace_3D(chip_ics)
    # monitor.report_core_power_var()
    # flp_generator = floorplan_generator(spec_info, connectivity)
    # flp_generator.gen_floorplan(chip_ics=0.000,xx=acc_xlen,yy=acc_ylen)
    # flp_generator.run_hotspot(network.net_name)

    flp_generator_3d.run_hotspot(draw_fig=True)
    max_temp = find_hotpoint(file='./outputs/gcc.grid.steady')


def argparser():
    ''' Argument parser. '''

    ap = argparse.ArgumentParser()

    ap.add_argument('net', nargs='+',
                    help='network name, should be a .py file under examples, could be a list', default=['toy_net'])

    ap.add_argument('-b', '--batch', type=int, default=1,
                    help='batch size')
    ap.add_argument('-w', '--word', type=int, default=8,
                    help='word size in bits')

    return ap


if __name__ == '__main__':
    layer_stats(argparser().parse_args())

