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
    ap.add_argument('-hp', '--hotspot_path', type=str, default='../../HotSpot',
                    help='the hotspot path')
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
hotspot_path = arg.hotspot_path
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
                    ,[i for i in range(1,16)], [i for i in range(1,16)], [i for i in range(1,6)], [i for i in range(4,17)], [4]]
elif isRunBaseline1:
    design_space = [[i for i in range(1,9)],[i for i in range(1,9)],[i for i in range(1,9)],[i for i in range(1,9)], [i for i in range(0,11)]
                    ,[i for i in range(1,16)], [i for i in range(1,16)], [i for i in range(1,6)], [i for i in range(4,17)], [4]]
else:
    design_space = [[i for i in range(1,9)],[i for i in range(1,9)],[i for i in range(1,9)],[i for i in range(1,9)], [i for i in range(0,11)]
                    ,[i for i in range(1,16)], [i for i in range(1,16)], [i for i in range(1,6)], [i for i in range(4,17)], [4]]


bounds = torch.tensor([[x[0] for x in design_space],
                    [x[-1] for x in design_space]])
dim = len(design_space)
lb, ub =bounds
bo_batch_size = 8
max_cholesky_size = float("inf")  # Always use Cholesky

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
    cost = EDY_cost + (maxT - peak_temp) * 0.05
    # print(f'E: {energy} D: {delay}, Yield: {die_yield}. The EDYP cost is {EDY_cost}, Cost is {cost}')
    return cost

def eval_objective(x, maxT, chiplet_sim_dict):
    """This is a helper function we use to unnormalize and evalaute a point"""
    unnorm = unnormalize(x, bounds)
    return target_function(unnorm, maxT, chiplet_sim_dict)

def c1(x, max_area, chiplet_sim_dict):
    sys_info = param_regulator(x)
    # evaluator = chiplet_evaluator(nn_wld, interposer_cons=0.03, hotspot_path='../../HotSpot', sim_path='../tmp', sys_info= sys_info)
    evaluator = chiplet_sim_dict[tuple(sys_info)]
    area = evaluator.evaluate_area()
    return  area - max_area

def eval_c1(x, max_area, chiplet_sim_dict):
    """This is a helper function we use to unnormalize and evalaute a point"""
    unnorm = unnormalize(x, bounds)
    return c1(unnorm, max_area, chiplet_sim_dict)

def c2(x, max_temp ,chiplet_sim_dict):
    sys_info = param_regulator(x)
    evaluator = chiplet_sim_dict[tuple(sys_info)]
    peak_temp = evaluator.evaluate_thermal()
    return peak_temp - max_temp

def eval_c2(x,max_temp, chiplet_sim_dict):
    """This is a helper function we use to unnormalize and evalaute a point"""
    unnorm = unnormalize(x, bounds)
    return c2(unnorm, max_temp, chiplet_sim_dict)

@dataclass
class ScboState:
    dim: int
    batch_size: int
    length: float = 1
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    # failure_tolerance: int = float("nan")  # Note: Post-initialized
    failure_tolerance: int = 10  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10 # Note: The original paper uses 3
    best_value: float = -float("inf")
    best_constraint_values: Tensor = torch.ones(2, **tkwargs) * torch.inf
    restart_triggered: bool = False

    # def __post_init__(self):
    #     self.failure_tolerance = math.ceil(max([40.0 / self.batch_size,  self.dim]))


def update_tr_length(state: ScboState):
    # Update the length of the trust region according to
    # success and failure counters
    # (Just as in original TuRBO paper)
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:  # Restart when trust region becomes too small
        state.restart_triggered = True

    return state

def get_best_index_for_batch(Y: Tensor, C: Tensor):
    """Return the index for the best point."""
    is_feas = (C <= 0).all(dim=-1)
    if is_feas.any():  # Choose best feasible candidate
        score = Y.clone()
        score[~is_feas] = -float("inf")
        return score.argmax()
    return C.clamp(min=0).sum(dim=-1).argmin()


def update_state(state, Y_next, C_next):
    """Method used to update the TuRBO state after each step of optimization.

    Success and failure counters are updated according to the objective values
    (Y_next) and constraint values (C_next) of the batch of candidate points
    evaluated on the optimization step.

    As in the original TuRBO paper, a success is counted whenver any one of the
    new candidate points improves upon the incumbent best point. The key difference
    for SCBO is that we only compare points by their objective values when both points
    are valid (meet all constraints). If exactly one of the two points being compared
    violates a constraint, the other valid point is automatically considered to be better.
    If both points violate some constraints, we compare them inated by their constraint values.
    The better point in this case is the one with minimum total constraint violation
    (the minimum sum of constraint values)"""

    # Pick the best point from the batch
    best_ind = get_best_index_for_batch(Y=Y_next, C=C_next)
    y_next, c_next = Y_next[best_ind], C_next[best_ind]

    if (c_next <= 0).all():
        # At least one new candidate is feasible
        improvement_threshold = state.best_value + 1e-3 * math.fabs(state.best_value)
        if y_next > improvement_threshold or (state.best_constraint_values > 0).any():
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = c_next
        else:
            state.success_counter = 0
            state.failure_counter += 1
    else:
        # No new candidate is feasible
        total_violation_next = c_next.clamp(min=0).sum(dim=-1)
        total_violation_center = state.best_constraint_values.clamp(min=0).sum(dim=-1)
        if total_violation_next < total_violation_center:
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = c_next
        else:
            state.success_counter = 0
            state.failure_counter += 1

    # Update the length of the trust region according to the success and failure counters
    state = update_tr_length(state)
    return state

state = ScboState(dim=dim, batch_size=bo_batch_size)
print(state)

def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    C,  # Constraint values
    batch_size,
    n_candidates,  # Number of candidates for Thompson sampling
    constraint_model,
    sobol: SobolEngine,
):
    if not (X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))):
        print(X)
        print(Y)
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    # Create the TR bounds
    best_ind = get_best_index_for_batch(Y=Y, C=C)
    x_center = X[best_ind, :].clone()
    tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)

    # Thompson Sampling w/ Constraints (SCBO)
    dim = X.shape[-1]
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    # Create a perturbation mask
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

    # Create candidate points from the perturbations and the mask
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # Sample on the candidate points using Constrained Max Posterior Sampling
    constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
        model=model, constraint_model=constraint_model, replacement=False
    )
    with torch.no_grad():
        X_next = constrained_thompson_sampling(X_cand, num_samples=batch_size)

    return X_next

def random_ted(size, design_space, verbose=False):
    from random import randint
    from sklearn.gaussian_process.kernels import RBF
    K = list(itertools.product(*design_space))
    m = size # init training set size
    Nrted = 59 # according to original paper
    u = 0.1 # according to original paper
    length_scale = 0.1 # according to original paper

    f = RBF(length_scale=length_scale)

    def F_kk(K):
        dis_list = []
        for k_i in K:
            for k_j in K:
                dis_list.append(f(np.atleast_2d(k_i), np.atleast_2d(k_j)))
        return np.array(dis_list).reshape(len(K), len(K))

    K_tilde = []
    for i in range(m):
        M = [K[randint(0,len(K)-1)] for _ in range(Nrted)]
        M = M + K_tilde
        F = F_kk(M)
        if verbose: print(F)
        denoms=[F[-i][-i] + u for i in range(len(K_tilde))]
        for i in range(len(denoms)):
            for j in range(len(M)):
                for k in range(len(M)):
                    F[j][k] -= (F[j][i] * F[k][i]) / denoms[i]
        if verbose: print('----------------------------\n', F)
        assert len(M) == F.shape[0]
        k_i = M[np.argmax([np.linalg.norm(F[i])**2 / (F[i][i] + u) for i in range(len(M))])] # find i that maximaize norm-2(column i of F)
        K_tilde.append(list(k_i))
    if verbose: print(K_tilde)
    return K_tilde

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

def process_one(x, bounds,hotspot_path, thermal_map, isRunBaseline1, isRunBaseline2, wkld_idpdt):
    # 每个进程自己的临时工作目录
    # work_dir = tempfile.mkdtemp(prefix=f"chiplet_{uuid.uuid4().hex}_")
    work_dir = creat_sim_tmpdir()
    try:
        sys_info = param_regulator(unnormalize(x, bounds))
        evaluator = chiplet_evaluator(
            hotspot_path=hotspot_path,
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

def parallel_run(train_X, bounds, hotspot_path, thermal_map, isRunBaseline1=False, isRunBaseline2=False, wkld_idpdt=None, max_workers=None):
    chiplet_eval_dict = {}
    worker = partial(
        process_one,
        hotspot_path = hotspot_path,
        bounds=bounds,
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


n_init = 32
seed = 1234
log_path = '.'
K_tilde = random_ted(n_init, design_space)
K_tilde = legalization(K_tilde, isRunBaseline2, isRunBaseline1)
readable_init = [param_regulator(x) for x in K_tilde]
print('init_designs', readable_init)
init_designs =torch.stack([normalize(torch.tensor(x, dtype=dtype), bounds) for x in K_tilde])
assert len(init_designs) == n_init

train_X = init_designs
# print(f'init : {init_designs}')
# print(f'legal: {train_X}')
# chiplet_eval_dict = {}

# ## run simulation 
# for x in train_X:
#     sys_info = param_regulator(unnormalize(x, bounds))
#     evaluator = chiplet_evaluator( hotspot_path='../../HotSpot', sim_path=sim_path, sys_info= sys_info, thermal_map=thermal_map, baseline1=isRunBaseline1, baseline2=isRunBaseline2, wkld_idpdt=wkld_idpdt)
#     evaluator.generate_hardware()
#     _, _ ,_ =evaluator.evaluate()
#     chiplet_eval_dict[tuple(sys_info)] = evaluator

chiplet_eval_dict = parallel_run(train_X,bounds,hotspot_path, thermal_map, isRunBaseline1, isRunBaseline2,wkld_idpdt,max_workers=4)

train_Y = torch.tensor([eval_objective(x, max_temp, chiplet_eval_dict) for x in train_X], **tkwargs).unsqueeze(-1)
C1 = torch.tensor([eval_c1(x, max_area, chiplet_eval_dict) for x in train_X], **tkwargs).unsqueeze(-1)
C2 = torch.tensor([eval_c2(x, max_temp, chiplet_eval_dict) for x in train_X], **tkwargs).unsqueeze(-1)
chiplet_eval_dict.clear()

reward = [train_Y[i]*((C1[i]<=0).type_as(train_Y[i])*((C2[i]<=0).type_as(train_Y[i]))) for i in range(n_init)]

with open(os.path.join(log_path, "details.txt"),"a") as f: 
    for i in range(n_init):
        f.write("{} - {} reward:{}\n".format(i, readable_init[i], float(reward[i])))
best_init = float(max(reward))
print('best of init EDPY: ', abs(best_init))
with open(os.path.join(log_path, "details.txt"),"a") as f: 
    f.write('best of init EDYP:{}\n'.format(abs(best_init)))

state = ScboState(dim, batch_size=bo_batch_size)
N_CANDIDATES = min(5000, max(3000, 200 * dim))
sobol = SobolEngine(dim, scramble=True, seed=1)

fit_arg={"max_attempts": 20}
def get_fitted_model(X, Y):
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
    )
    model = SingleTaskGP(
        X,
        Y,
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        fit_gpytorch_mll(mll)

    return model

count = n_init

while not state.restart_triggered and count < 1000:  # Run until TuRBO converges
    count+=1
    # Fit GP models for objective and constraints
    model = get_fitted_model(train_X, train_Y)
    c1_model = get_fitted_model(train_X, C1)
    c2_model = get_fitted_model(train_X, C2)

    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        X_next = generate_batch(
            state=state,
            model=model,
            X=train_X,
            Y=train_Y,
            C=torch.cat((C1, C2), dim=-1),
            batch_size=bo_batch_size,
            n_candidates=N_CANDIDATES,
            constraint_model=ModelListGP(c1_model, c2_model),
            sobol=sobol,
        )
    ## run simulation for the next batch
    # print(f'X_next before: {unnormalize(X_next, bounds)}')
    X_next = legalization(unnormalize(X_next, bounds), isRunBaseline2, isRunBaseline1)
    # print(f'X_next after: {X_next}')
    X_next = normalize(X_next, bounds)
    # for x in X_next:
    #     sys_info = param_regulator(unnormalize(x, bounds))
    #     evaluator = chiplet_evaluator( hotspot_path='../../HotSpot', sim_path=sim_path, sys_info= sys_info, thermal_map=thermal_map,baseline1=isRunBaseline1, baseline2=isRunBaseline2,wkld_idpdt=wkld_idpdt)
    #     evaluator.generate_hardware()
    #     _, _ ,_ =evaluator.evaluate()
    #     chiplet_eval_dict[tuple(sys_info)] = evaluator
    chiplet_eval_dict = parallel_run(X_next,bounds,hotspot_path,thermal_map, isRunBaseline1, isRunBaseline2,wkld_idpdt,max_workers=4)

    # Evaluate both the objective and constraints for the selected candidaates
    Y_next = torch.tensor([eval_objective(x, max_temp,chiplet_eval_dict) for x in X_next], dtype=dtype, device=device).unsqueeze(-1)
    C1_next = torch.tensor([eval_c1(x, max_area, chiplet_eval_dict) for x in X_next], dtype=dtype, device=device).unsqueeze(-1)
    C2_next = torch.tensor([eval_c2(x, max_temp, chiplet_eval_dict) for x in X_next], dtype=dtype, device=device).unsqueeze(-1)
    C_next = torch.cat([C1_next, C2_next], dim=-1)
    chiplet_eval_dict.clear()

    # Update TuRBO state
    state = update_state(state=state, Y_next=Y_next, C_next=C_next)
    # Append data. Note that we append all data, even points that violate
    # the constraints. This is so our constraint models can learn more
    # about the constraint functions and gain confidence in where violations occur.
    train_X = torch.cat((train_X, X_next), dim=0)
    train_Y = torch.cat((train_Y, Y_next), dim=0)
    C1 = torch.cat((C1, C1_next), dim=0)
    C2 = torch.cat((C2, C2_next), dim=0)
    # Print current status. Note that state.best_value is always the best
    # objective value found so far which meets the constraints, or in the case
    # that no points have been found yet which meet the constraints, it is the
    # objective value of the point with the minimum constraint violation.
    if (state.best_constraint_values <= 0).all():
        print(f"{len(train_X)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}")
    else:
        violation = state.best_constraint_values.clamp(min=0).sum()
        print(
            f"{len(train_X)}) No feasible point yet! Smallest total violation: "
            f"{violation:.2e}, TR length: {state.length:.2e}"
        )
    # torch.save(train_X, "scbo/train_X.pt")
    # torch.save(train_Y, "scbo/train_Y.pt")
    # torch.save(C1, "scbo/C1.pt")
    # torch.save(C2, "scbo/C2.pt")
