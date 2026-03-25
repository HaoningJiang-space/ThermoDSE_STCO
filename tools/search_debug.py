
import argparse
import sys

import concurrent.futures
import tempfile
import uuid
import shutil
import os
from functools import partial
# from search_util import process_one

sys.path.append('../')


from core.chiplet_eva import chiplet_evaluator

def argparser():
    ''' Argument parser. '''

    ap = argparse.ArgumentParser()

    ap.add_argument('-m', '--map', type=int, default=0,
                    help='is this Search including map')
    ap.add_argument('-tm', '--thermal_aware_map', type=int, default=1,
                    help='1: thermal awawre mapping, 0: data aware mapping')
    ap.add_argument('-b1', '--baseline1', type=int, default=0,
                    help='chiplet-gym, not data buffering')
    ap.add_argument('-b2', '--baseline2', type=int, default=0,
                    help='TESA, not NoC/NoP, ideal scheduling')
    ap.add_argument('-b3', '--baseline3', type=int, default=0,
                help='TESA, not NoC/NoP, none ideal scheduling.')
    ap.add_argument('-sp', '--sim_path', type=str, default='../tmp',
                    help='thermal simulation path')
    ap.add_argument('-wi', '--wkld_idpdt', type=int, default=0,
                    help='peak temp. is max(peak_temp) ')
    return ap

arg = argparser().parse_args()
isSearchMapping = arg.map
isRunBaseline1 = arg.baseline1
isRunBaseline2 = arg.baseline2
isRunBaseline3 = arg.baseline3
sim_path = arg.sim_path
wkld_idpdt = arg.wkld_idpdt
thermal_map = arg.thermal_aware_map
if isRunBaseline1:
    target_arch = 'chiplet-gym'
elif isRunBaseline2:
    target_arch = 'TESA'
else:
    target_arch = 'scalable MCM'
print(f'START SCBO TWO STAGE SEARCH, area constriant: 300 mm^2, Peak temp. constraint: 350K. Targe arch:{target_arch}\n')

def target_function(xx, yy, cx, cy, ci , hsa, wsa, ubf, nop_bw, dram_bw):
    chipletX = int(xx)
    chipletY = int(yy)
    chipletCx = int(cx) if cx < chipletX else int(chipletX)
    chipletCy = int(cy) if cy < chipletY else int(chipletY)
    chipletIntvl = 0.0005 + int(ci) * 0.0003
    h_sa = 16 * int(hsa)  
    w_sa = 16 * int(wsa)
    ubuf_size = 128 * 2** int(ubf) *1024
    nop_bw = 16*nop_bw
    dram_bw = 128
    sys_info = [chipletX, chipletY, chipletCx, chipletCy, chipletIntvl, h_sa, w_sa, ubuf_size, nop_bw, dram_bw]
    print(f'sys_info:{sys_info}, cost:')
    evaluator = chiplet_evaluator(hotspot_path='../../HotSpot', sim_path=sim_path, sys_info= sys_info,thermal_map=thermal_map, baseline1=isRunBaseline1, baseline2=isRunBaseline2,  baseline3=isRunBaseline3, wkld_idpdt=wkld_idpdt)
    evaluator.generate_hardware()
    delay, energy, die_yield = evaluator.evaluate(draw_fig=True)
    # print(f'{EDY_cost}\n')
    compute_die_area = evaluator.get_compute_die_area()
    IO_die_area = evaluator.get_IO_die_area()
    print(f'Area of Computing die: {compute_die_area}, IO die: {IO_die_area}')
    EDY_cost = - delay * energy / die_yield
    print(f'E: {energy} D: {delay}, Yield: {die_yield}. The EDYP cost is {EDY_cost}')
    return sys_info, EDY_cost




if __name__ == '__main__':
    arg = argparser().parse_args()
    # nn_wld = arg.net[0]
    # batch = arg.batch
    # sys_info, _ =target_function(6, 6, 6, 6, 10, 10, 9, 1, 5, 4) 
    # sys_info, _ =target_function(5, 2, 5, 2, 0, 12, 8, 6, 4, 4)
    # target_function(6, 2, 6, 2, 0, 8, 16, 5, 8, 4)  # TESA non-ideal scheduler
    # target_function(8, 2, 8, 2, 0, 9, 8, 5, 6, 4)  # TESA ideal scheduler
    # target_function(2, 2, 2, 2, 8, 10, 10, 6, 13, 4)  # TPU 2X2
    # target_function(4, 4, 4, 1, 3, 13, 8, 3, 13, 4)  # opt design
    # target_function(6, 6, 2, 2, 10, 2, 4, 2, 6, 4)  # TPU 2X2
    # target_function(4, 5, 2, 1, 3, 8, 11, 3, 15, 4)  # scbo 348 300  cost 280
    # target_function(5, 7, 2, 2, 1, 11, 6, 2, 14, 4)  # scbo 348 300  cost 280
    # target_function(4, 4, 2, 4, 4, 9, 7, 3, 12, 4)  # scbo 358 400  cost 332
    target_function(5,4,1,2,6,13, 7, 3, 15,4)  # scbo best 
    # target_function(4,4,1,2,6,9, 7, 3, 15,4)  # scbo best 
    # target_function(4,4,4,4,2, 10, 10, 3, 6, 4)  # Chiplet-Gym
    # target_function(3,4,1,2,3, 12,14, 2, 10, 4) # scbo archs bert 


    # target_function(5,4,1,2,3,9,10,3,9,4)   # scbo+arch-s 348 300, inital temp for SA 200, peak temp 347.8, EDPY 237.6
    # target_function(7,3,1,1,3,9,8,2,9,4)   # SA+arch-s 348 300, inital temp for SA 300, peak temp 347.4, EDPY 290.3
    # target_function(5,5,1,1,3,10,6,2,7,4)   # SA+arch-s 348 300, inital temp for SA 400, peak temp 346.1, EDPY 283.7
    # target_function(5,6,1,2,2,10,5,2,9,4)
    # target_function(3,2,3,2, 0, 10, 13, 5,5, 4)  # TESA
    # target_function(8,2,8,2,0,8,8,5,15,4)   # TESA* 348 300, non-ideal scheduler: peak temp 336.2, ideal scheduler: 343.6
    # target_function(7,2,7,2,0,8,8,5,15,4)   # TESA* 348 300, round 2
    # target_function(1,1,1,1,2,1,2,1,1,4)

    ## motivated example 
    # target_function(6, 6, 6, 6, 4, 4, 1, 2, 8, 4)  # simba 
    # target_function(6, 6, 3, 3, 6, 4, 2, 3, 6, 4)
    # target_function(4, 4, 2, 2, 10, 4, 4, 4, 8, 4) 

    # process_one(sys_info,sim_path=sim_path,thermal_map=thermal_map, baseline1=isRunBaseline1, baseline2=isRunBaseline2, wkld_idpdt=wkld_idpdt)
    ## hardware search space
    # pbounds = {'cs':(1,4+1), 'cc':(1,6+1), 'ci': (1, 25+1), 'hsa':(1, 8+1), 'wsa': (1,8+1), 'ubf':(0, 6)}
    # optimizer = BayesianOptimization(
    #     f = target_function,
    #     pbounds = pbounds,
    #     random_state = 1234,
    #     verbose = 0
    # )
    # optimizer.probe(params={'cs':4, 'cc':6, 'ci': 25, 'hsa':8, 'wsa': 8, 'ubf':6}, lazy=True,)
    # optimizer.maximize(init_points= 1, n_iter= 0)




    