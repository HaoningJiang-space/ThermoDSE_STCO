import argparse
import os, sys
sys.path.append('../')

import numpy as np
from numpy.random import rand, uniform
import random
from core.chiplet_eva import chiplet_evaluator


# def argparser():
#     ''' Argument parser. '''

#     ap = argparse.ArgumentParser()

#     ap.add_argument('net', nargs='+',
#                     help='network name, should be a .py file under examples, could be a list', default=['toy_net'])

#     ap.add_argument('-b', '--batch', type=int, default=1,
#                     help='batch size')
#     ap.add_argument('-w', '--word', type=int, default=8,
#                     help='word size in bits')

#     return ap

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
                    help='TESA, not NoC/NoP')
    ap.add_argument('-b3', '--baseline3', type=int, default=0,
            help='TESA, not NoC/NoP, none ideal scheduling.')
    ap.add_argument('-hp', '--hotspot_path', type=str, default='../../HotSpot',
                    help='the hotspot path')
    ap.add_argument('-sp', '--sim_path', type=str, default='../tmp',
                    help='thermal simulation path')
    ap.add_argument('-wi', '--wkld_idpdt', type=int, default=0,
                    help='peak temp. is max(peak_temps of each nns) ')
    ap.add_argument('-maxT', '--max_temp', type=int, default=348,
                help='peak temperature constraints ')
    ap.add_argument('-maxA', '--max_area', type=int, default=300,
            help='max area constraints (mm^2)')
    return ap

def action_refined(x):
    action_new = x.copy()
    action_new[2] = x[2] if x[2] < x[0] else x[0]
    action_new[3] = x[3] if x[3] < x[1] else x[1]
    # action_new[-1] = x[-1] if int (x[0]) * int(x[1]) > int(x[-1]) else int(x[0])* int(x[1]) - 1
    return action_new

def param_regulator(x):
    xx = x[0] # cores in x dim
    yy = x[1] # cores in y dim
    cx = x[2] # chiplet cut
    cy = x[3]
    ci = x[4] # chiplet interval space
    hsa = x[5] # height of systolic array / matrix unit
    wsa = x[6] # width of systolic array / matrix unit
    ubf = x[7] # uniformed buffer size
    nop = x[8]
    dram = x[9]
    xx, yy = int(xx), int(yy)
    cx, cy = int(cx), int(cy) 
    chipletIntvl = 0.0005 + int(ci) * 0.0003
    h_sa = 16 * int(hsa)  
    w_sa = 16 * int(wsa)
    ubuf_size = 128 * 2** int(ubf) *1024
    nop_bw = 16 * int(nop)
    dram_bw = 32 * 4
    # force_closed_core_wld0 = int(fcc_0)
    # force_closed_core_wld1 = int(fcc_1)
    # force_closed_core_wld2 = int(fcc_2)
    sys_info = [xx, yy,cx, cy, chipletIntvl, h_sa, w_sa, ubuf_size, nop_bw, dram_bw]
    return sys_info

def target_function(x, hotspot_path, max_temp, max_area , sim_path, thermal_map,baseline1, baseline2, baseline3, wkld_idpdt):
    sys_info = param_regulator(x)
    evaluator = chiplet_evaluator( hotspot_path=hotspot_path, sim_path=sim_path, sys_info= sys_info,thermal_map=thermal_map,baseline1=baseline1, baseline2=baseline2,baseline3=baseline3, wkld_idpdt=wkld_idpdt)
    evaluator.generate_hardware()
    delay, energy, die_yield = evaluator.evaluate()
    delay, energy, die_yield = evaluator.evaluate_edyp()
    EDPY_cost = - energy * delay / die_yield
    peak_temp = evaluator.evaluate_thermal()
    area = evaluator.evaluate_area()
    if peak_temp > max_temp or area > max_area:
        EDPY_cost = -1e5
    return EDPY_cost


arg = argparser().parse_args()
isSearchMapping = arg.map
hotspot_path = arg.hotspot_path
isRunBaseline1 = arg.baseline1
isRunBaseline2 = arg.baseline2
isRunBaseline3 = arg.baseline3
sim_path = arg.sim_path
wkld_idpdt = arg.wkld_idpdt
thermal_map = arg.thermal_aware_map
max_temp = arg.max_temp
max_area = arg.max_area * 1e-6 # mm^2 -> m^2

if isRunBaseline1:
    target_arch = 'chiplet-gym'
elif isRunBaseline2 or isRunBaseline3:
    target_arch = 'TESA'
else:
    target_arch = 'scalable MCM'
print(f'START SCBO TWO STAGE SEARCH, simulation path {sim_path}, area constriant: {max_area} m^2, Peak temp. constraint: {max_temp} K. Targe arch:{target_arch}\n')

# parameter_space = np.array([[1, 17],[1, 9], [0, 25], [1, 9], [1, 9], [0, 7]])
if isRunBaseline2 or isRunBaseline3:
    if isSearchMapping == 0:
        design_space = [[1,8],[1,8],[8,9],[8,9], [0,10], [1,16], [1,16], [1,8], [4,16],[4,5]]
    else:
        design_space = [[1,8],[1,2],[8,9],[8,9], [0,10], [1,16], [1,16], [1,8], [4,16],[1,8]]
else:
    if isSearchMapping == 0:
        design_space = [[1,8],[1,8],[1,8],[1,8], [0,10], [1,16], [1,16], [1,8], [4,16],[4,5]]
    else:
        design_space = [[1,8],[1,8],[1,8],[1,8], [0,10], [1,16], [1,16], [2,8], [4,16],[1,8]]

parameter_space = np.array(design_space)
best_parameter = parameter_space[:, 0] + rand(len(parameter_space)) * (parameter_space[:, 1] - parameter_space[:, 0])
best_parameter = best_parameter.round()

parameter_after = action_refined(best_parameter)
best_edpy = target_function(parameter_after,hotspot_path, max_temp, max_area, sim_path ,thermal_map,isRunBaseline1, isRunBaseline2,isRunBaseline3, wkld_idpdt)
curr_param, curr_edpy = best_parameter, best_edpy
num_trials = 4000
tmp = 200
step_size = 3

for i in range(num_trials):
    candidate_param_before = curr_param + uniform(-1, 1, len(parameter_space)) * step_size
    candidate_param_before = [int(i) for i in candidate_param_before]
    for j in range(0, len(candidate_param_before)):
        if candidate_param_before[j] > parameter_space[j, 1]:
            candidate_param_before[j] = parameter_space[j, 1]
        if candidate_param_before[j] < parameter_space[j, 0]:
            candidate_param_before[j] = parameter_space[j, 0]
    candidate_param = action_refined(candidate_param_before)
    candidate_edpy = target_function(candidate_param, hotspot_path, max_temp, max_area, sim_path,thermal_map,isRunBaseline1,isRunBaseline2,isRunBaseline3, wkld_idpdt)
    if candidate_edpy > best_edpy:
        best_parameter, best_edpy = candidate_param, candidate_edpy
    
    t = tmp /float(i+1)
    if candidate_edpy > curr_edpy or rand() < t:
        curr_param, curr_edpy = candidate_param, candidate_edpy

print(f'Best EDPY:{best_edpy}, sys info:{param_regulator(best_parameter)}')