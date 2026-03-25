import argparse

import math
import os, sys
import warnings
from dataclasses import dataclass
import itertools
import numpy as np

import concurrent.futures
import tempfile
import uuid
import shutil
import os
from functools import partial
from search_util import *

sys.path.append('../')

import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.fit import fit_gpytorch_mll
# Constrained Max Posterior Sampling s a new sampling class, similar to MaxPosteriorSampling,
# which implements the constrained version of Thompson Sampling described in [1].
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize, normalize

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
                    help='TESA, not NoC/NoP')
    ap.add_argument('-sp', '--sim_path', type=str, default='../tmp',
                    help='thermal simulation path')
    ap.add_argument('-wi', '--wkld_idpdt', type=int, default=0,
                    help='peak temp. is max(peak_temp) ')
    ap.add_argument('-maxT', '--max_temp', type=int, default=348,
            help='peak temperature constraints ')
    ap.add_argument('-maxA', '--max_area', type=int, default=200,
            help='max area constraints (mm^2)')
    return ap

arg = argparser().parse_args()
isSearchMapping = arg.map
isRunBaseline1 = arg.baseline1
isRunBaseline2 = arg.baseline2
sim_path = arg.sim_path
wkld_idpdt = arg.wkld_idpdt
thermal_map = arg.thermal_aware_map
max_temp = arg.max_temp
max_area = arg.max_area * 1e-6 # mm^2 -> m^2

if isRunBaseline1:
    target_arch = 'chiplet-gym'
elif isRunBaseline2:
    target_arch = 'TESA'
else:
    target_arch = 'scalable MCM'
print(f'START SCBO TWO STAGE SEARCH, simulation path {sim_path}, area constriant: {max_area} m^2, Peak temp. constraint: {max_temp} K. Targe arch:{target_arch}\n')

device = torch.device("cpu")
dtype = torch.double
tkwargs = {"device": device, "dtype": dtype}

if isRunBaseline2:
    design_space = [[i for i in range(1,9)],[i for i in range(1,3)],[i for i in range(1,9)],[i for i in range(1,3)], [i for i in range(0,11)]
                    ,[i for i in range(1,16)], [i for i in range(1,16)], [i for i in range(1,7)], [i for i in range(4,17)], [4]]
elif isRunBaseline1:
    design_space = [[i for i in range(1,9)],[i for i in range(1,9)],[i for i in range(1,9)],[i for i in range(1,9)], [i for i in range(0,11)]
                    ,[i for i in range(1,16)], [i for i in range(1,16)], [i for i in range(1,7)], [i for i in range(4,17)], [4]]
else:
    design_space = [[i for i in range(1,9)],[i for i in range(1,9)],[i for i in range(1,9)],[i for i in range(1,9)], [i for i in range(0,11)]
                    ,[i for i in range(1,16)], [i for i in range(1,16)], [i for i in range(1,7)], [i for i in range(4,17)], [4]]



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
    dram_bw = 32 * int(dram)
    sys_info = [xx, yy,cx, cy, chipletIntvl, h_sa, w_sa, ubuf_size, nop_bw, dram_bw]
    return sys_info

def target_function(x, maxT, chiplet_sim_dict):
    sys_info = param_regulator(x)
    # print(f'sys_info:{sys_info}, {chiplet_sim_dict}')
    evaluator = chiplet_sim_dict[tuple(sys_info)]
    delay, energy, die_yield = evaluator.evaluate_edyp()
    peak_temp = evaluator.evaluate_thermal()
    # print(f'{EDY_cost}\n')
    EDY_cost = - energy * delay / die_yield
    cost = EDY_cost + (maxT - peak_temp) * 2
    # print(f'E: {energy} D: {delay}, Yield: {die_yield}. The EDYP cost is {EDY_cost}, Cost is {cost}')
    return cost



def legalization(X, isRunBaseline2, isRunBaseline1):
    ## the chiplet cut should be less than the chipletsize
    ## the forced closed core should be less than total core
    for i, x in enumerate(X):
        if x[2] > x[0] or isRunBaseline2 or isRunBaseline1:
            X[i][2] = X[i][0]  #  xcut > xx
        if x[3] > x[1] or isRunBaseline2 or isRunBaseline1:
            X[i][3] = X[i][1]  # ycut > yy
        # if int (x[0]) **2 <= int(x[6]):
        #     X[i][6] = int(x[0])**2 - 1
        # if int (x[0]) **2 <= int(x[7]):
        #     X[i][7] = int(x[0])**2 - 1
        # if int (x[0]) * int(x[1]) <= int(x[-1]):
        #     X[i][-1] = int(x[0])* int(x[1]) - 1
    return X

def process_one(x, thermal_map, isRunBaseline1, isRunBaseline2, wkld_idpdt):
    # 每个进程自己的临时工作目录
    # work_dir = tempfile.mkdtemp(prefix=f"chiplet_{uuid.uuid4().hex}_")
    work_dir = creat_sim_tmpdir()
    try:
        sys_info = param_regulator(x)
        evaluator = chiplet_evaluator(
            hotspot_path='../../HotSpot',
            sim_path=work_dir,
            sys_info=sys_info,
            thermal_map=thermal_map,
            baseline1=isRunBaseline1,
            baseline2=isRunBaseline2,
            wkld_idpdt=wkld_idpdt,
        )
        evaluator.generate_hardware()
        evaluator.evaluate()
        return tuple(sys_info), evaluator
    finally:
        # 如果不需要保留临时文件，可清理
        shutil.rmtree(work_dir, ignore_errors=True)

def parallel_run(train_X, thermal_map, isRunBaseline1=False, isRunBaseline2=False, wkld_idpdt=None, max_workers=None):
    chiplet_eval_dict = {}
    worker = partial(
        process_one,
        thermal_map=thermal_map,
        isRunBaseline1=isRunBaseline1,
        isRunBaseline2=isRunBaseline2,
        wkld_idpdt=wkld_idpdt
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, x): x for x in train_X}
        for future in concurrent.futures.as_completed(futures):
            key, evaluator = future.result()
            chiplet_eval_dict[key] = evaluator
    return chiplet_eval_dict

center_points = [5,4,1,2,3,9,10,3,9,4]

xx_lb = 4
xx_ub = 6
yy_lb = 3
yy_ub = 5
ci_lb = 6
ci_ub = 9
hsa_lb = 8
hsa_ub = 14
wsa_lb = 8
wsa_ub = 14
ubf_lb = 2
ubf_ub = design_space[7][-1]
nop_list = [8, 10, 11, 12, 13 ,14,15, 16]
print(f'XX : start from: {xx_lb}, end at:{xx_ub}')
print(f'YY : start from: {yy_lb}, end at:{yy_ub}')
print(f'ci : start from: {ci_lb}, end at:{ci_ub}')
print(f'hsa: start from: {hsa_lb}, end at:{hsa_ub}')
print(f'wsa: start from: {wsa_lb}, end at:{wsa_ub}')
print(f'ubf: start from: {ubf_lb}, end at:{ubf_ub}')
print(f'nop: start from: 6, end at:15')
# total = (xx_ub-xx_lb + 1) * (yy_ub-yy_lb + 1) * (ci_ub-ci_lb + 1) * (hsa_ub-hsa_lb + 1)  * (wsa_ub-wsa_lb + 1) * (ubf_ub-ubf_lb + 1) * len(nop)  
# print(f'Total design points: {total}')

for xx in range(xx_lb,xx_ub , 1):
    for yy in range(yy_lb,yy_ub, 1):
        for cx in range(1, min(4,xx), 1):
            for cy in range(1, min(4,yy), 1):
                for ci in range(ci_lb, ci_ub,1):
                    for hsa in range(hsa_lb, hsa_ub, 1):
                        for wsa in range(wsa_lb, wsa_ub, 1):
                            for ubf in range(ubf_lb, ubf_ub, 1):
                                train_X = []
                                for nop in nop_list:
                                    train_X.append([xx, yy, cx, cy, ci, hsa, wsa, ubf, nop, 4])
                                chiplet_eval_dict = parallel_run(train_X,thermal_map, isRunBaseline1, isRunBaseline2,wkld_idpdt,max_workers=len(nop_list))
                                for sys_info, eva in chiplet_eval_dict.items():
                                    area = eva.evaluate_area()
                                    peak_temp = eva.evaluate_thermal()
                                    delay, energy, die_yield = eva.evaluate_edyp()
                                    print(f'sys_info:{sys_info}, Area:{area}, peak_temp:{peak_temp},EDYP:{delay * energy / die_yield}, Delay:{delay}, energy:{energy}, Yield:{die_yield}.')