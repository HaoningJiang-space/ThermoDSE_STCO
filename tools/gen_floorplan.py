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


KILO = 1024.
MILLION = 1024.*1024.


def layer_stats(args):
    # Hardware initalization assuming the clock frequency is 1GHz
    AREA_SCALE_10 = 11.667
    E_SCALE_10 = 2.467 
    AREA_SCALE_7 = 31.818
    E_SCALE_7 = 3.364
    clock = 1e9
    acc_xlen = 6
    acc_ylen = 6
    xcut = 2
    ycut = 2
    smtxu = (32, 64)
    minSmtx_part = 4
    smtxu_grid = (16,16)
    svecu = smtxu[0]
    ubuf_cap =0.5 *1024 *KILO
    l0a_cap = l0b_cap = l1c_cap = 128 *KILO 
    l0c_cap = 32*KILO 
    noc_bw = 128
    nop_bw = 64
    # dram_bw = int(acc_xlen * acc_ylen * smtxu[0] * smtxu[1] *2 * clock *2 / 1e12)  # 1GB/s per TOPS (AL. 0.5, 1)
    hop_cost = 1.2 * 8   ## simba 0.82-1.75 pJ/bit
    noc_cost = 0.61 * 8
    dram_acc_cost = 7.5*8   ## SET/Aotmic HBM
    logic_density = 0.6
    dram_list = [(0,0), (0, acc_ylen -1), (acc_xlen + 1, 0), (acc_xlen + 1, acc_ylen - 1)]
    # for i in range(acc_ylen):
    #     dram_list.append((0, i))
    #     dram_list.append((acc_xlen + 1, i))
    dram_bw = 128 // len(dram_list) 
    ubuf_area, ubuf_rcost, ubuf_wcost = sram_setting_gen(ubuf_cap,1,1)
    l0a_area, l0a_rcost, l0a_wcost = sram_setting_gen(l0a_cap,1,1)
    l0b_area, l0b_rcost, l0b_wcost = sram_setting_gen(l0b_cap,1,1)
    l0c_area, l0c_rcost, l0c_wcost = regf_setting_gen(l0c_cap,1,1)
    l1c_area, l1c_rcost, l1c_wcost = sram_setting_gen(l1c_cap,1,1)
    mtxu_area, mtxu_cost = mtxu_settting_gen(smtxu, logic_density,1)
    vecu_area, vecu_cost = vecu_settting_gen(svecu, logic_density,1)
    ubuf = Onchip_buffer('ubuf', ubuf_cap, ubuf_rcost, ubuf_wcost,1, smtxu[0])
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

    chip_ics = args.ics
    flp_generator_3d = floorplan_generator_3D(spec_info)
    flp_generator_3d.gen_core_floorplan()
    sys_w, sys_y = flp_generator_3d.gen_sys_floorplan(w_const=0.020, h_const=0.020, ics=chip_ics,xx=acc_xlen,yy=acc_ylen, xcut=xcut, ycut=ycut)
    flp_generator_3d.run_hotspot()


def argparser():
    ''' Argument parser. '''

    ap = argparse.ArgumentParser()

    ap.add_argument('-ics', '--ics', type=float, default=0.001,
                    help='inter-chiplet space range from [0.001, 0.005]')

    return ap


if __name__ == '__main__':
    layer_stats(argparser().parse_args())

