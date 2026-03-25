import argparse
import os, sys
import gymnasium as gym
from gym import Env
from gym import spaces
import numpy as np
import random
import datetime
from stable_baselines3 import PPO, A2C, SAC, TD3, DQN, HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from typing import Callable

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
                    help='TESA, not NoC/NoP')
    ap.add_argument('-sp', '--sim_path', type=str, default='../tmp',
                    help='thermal simulation path')
    ap.add_argument('-wi', '--wkld_idpdt', type=int, default=0,
                    help='peak temp. is max(peak_temp) ')
    return ap

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
    dram = 4
    xx, yy = int(xx), int(yy)
    cx = int(cx) if xx > cx else xx
    cy = int(cy) if yy > cy else yy
    chipletIntvl = 0.0005 + int(ci) * 0.0003
    h_sa = 16 * int(hsa)  
    w_sa = 16 * int(wsa)
    ubuf_size = 128 * 2** int(ubf) *1024
    nop_bw = 16 * int(nop)
    dram_bw = 32 * int(dram)
    sys_info = [xx, yy,cx, cy, chipletIntvl, h_sa, w_sa, ubuf_size, nop_bw, dram_bw]
    return sys_info

def action_filter(actions, params, bounds):
    valid_actions = []
    valid_idxs = []
    actions_probs = []
    for i in range(len(params)):
        if params[i] < bounds[i]:
            valid_actions.append(actions[i])
            valid_idxs.append(i)
            actions_probs.append(-1)
        elif params[i] == bounds[i]:
            actions_probs.append(0)
        else:
            raise Exception('parameter {} exceeds upper bound: {} > {}'.format(i, params[i], bounds[i]))
    assert len(valid_idxs) > 0
    valid_sum = sum(valid_actions)
    if valid_sum == 0:
        valid_actions=[1 for i in valid_idxs]
        valid_sum = sum(valid_actions)
    assert valid_sum > 0
    valid_probs = [x/valid_sum for x in valid_actions]
    for idx, p in zip(valid_idxs, valid_probs):
        actions_probs[idx] = p
    assert all(x >= 0 for x in actions_probs), "probilities should not contain negative number"
    return actions_probs
    
class CustomEnv(Env):
    def __init__(self, process_id, sim_path, thermal_map, isRunBaseline1, isRunBaseline2, wkld_idpdt):
        self.process_id = process_id
        self.max_interposer_area = 0.0003       # unit m^2
        self.normlized_max_interposer_area = 1 # (Area/3)
        self.max_temperature = 348 # unit K
        self.normlized_max_temperature = 1 # (T-319)/30
        # self.action_space = spaces.Discrete(6)
        self.action_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))
        self.action_upper_bound = [8, 8, 8, 8, 10, 16, 16, 6, 16]
        self.obs_space_low = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.obs_space_high = np.array([self.normlized_max_interposer_area, self.normlized_max_temperature] + self.action_upper_bound)
        self.observation_space = spaces.Box(low = self.obs_space_low, high = self.obs_space_high)
        self.current_obs = None
        self.time_step = None

        self.sim_path = sim_path
        self.thermal_map = thermal_map
        self.isRunBaseline1 = isRunBaseline1
        self.isRunBaseline2 = isRunBaseline2
        self.wkld_idpdt = wkld_idpdt

    def reset(self):
        self.time_step = 0
        self.current_params = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.current_obs = np.concatenate((np.array(
            [0.07, 0.02]), self.current_params/np.array(self.action_upper_bound)))
        return self.current_obs

    def step(self, action):
        print("step:", self.time_step)
        """Takes action and returns next observation, reward done and optionally additional info"""
        done = False
        print(f'action before:{action} {self.current_params}')
        action_probs = action_filter(action, self.current_params, self.action_upper_bound)
        print(f'action_probs:{action_probs}')
        chosen_param = np.random.choice(range(len(self.current_params)), size=None,replace=None, p=action_probs)
        self.current_params[chosen_param]+=1
        print(f'action before:{self.current_params}')
        sys_info = param_regulator(self.current_params)
        print(f'sys info:{sys_info}')
        evaluator = chiplet_evaluator(hotspot_path='../../HotSpot', sim_path=self.sim_path, sys_info= sys_info, thermal_map=self.thermal_map,baseline1=self.isRunBaseline1, baseline2=self.isRunBaseline2,wkld_idpdt=self.wkld_idpdt)
        evaluator.generate_hardware()
        delay, energy, die_yield=evaluator.evaluate()
        EDPY_cost = - energy * delay / die_yield # 需要归一化
        peak_temp = evaluator.evaluate_thermal()
        area = evaluator.evaluate_area() 
        r_delay = delay/34
        r_energy = energy/50
        r = die_yield/(r_delay*r_energy)
        print(f'sys_info:{sys_info}, Area: {area*10000} cm^2, peak temp: {peak_temp}')
        # print(f'Area of Computing die: {compute_die_area}, IO die: {IO_die_area}')
        print(f'E: {energy} D: {delay}, Yield: {die_yield}, total area: {area}. The EDYP cost is {EDPY_cost}')
        with open(os.path.join("/home/jpengai/WorkSpace/power_project/thermal_project/accDSE/rl_opt/stats/archs", "Energy.txt"),"a") as f:
            f.write(str(energy)+"\n")
        with open(os.path.join("/home/jpengai/WorkSpace/power_project/thermal_project/accDSE/rl_opt/stats/archs", "delay.txt"),"a") as f:
            f.write(str(delay)+"\n")
        with open(os.path.join("/home/jpengai/WorkSpace/power_project/thermal_project/accDSE/rl_opt/stats/archs", "yield.txt"),"a") as f:
            f.write(str(die_yield)+"\n")
        with open(os.path.join("/home/jpengai/WorkSpace/power_project/thermal_project/accDSE/rl_opt/stats/archs", "temp.txt"),"a") as f:
            f.write(str(peak_temp)+"\n")
        with open(os.path.join("/home/jpengai/WorkSpace/power_project/thermal_project/accDSE/rl_opt/stats/archs", "area.txt"),"a") as f:
            f.write(str(area)+"\n")
        next_obs = np.concatenate((np.array([area * 10000 /3, (peak_temp-319)/30]), self.current_params/np.array(self.action_upper_bound)))
        reward = r - max(0, (peak_temp - self.max_temperature))

        self.time_step += 1
        if self.time_step > 50 or area >= self.max_interposer_area:
            done = True
            reward = 0
        self.current_obs = next_obs

        return self.current_obs, reward, done, {}

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass


def make_env(rank, seed, sim_path, thermal_map, isRunBaseline1, isRunBaseline2, wkld_idpdt) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = CustomEnv(rank, sim_path + "_" + str(rank), thermal_map, isRunBaseline1, isRunBaseline2, wkld_idpdt)
        # env.reset(seed=seed+rank)
        env.reset()
        return env

    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    arg = argparser().parse_args()
    isSearchMapping = arg.map
    isRunBaseline1 = arg.baseline1
    isRunBaseline2 = arg.baseline2
    sim_path_name = arg.sim_path
    wkld_idpdt = arg.wkld_idpdt
    thermal_map = arg.thermal_aware_map
    if isRunBaseline1:
        target_arch = 'chiplet-gym'
    elif isRunBaseline2:
        target_arch = 'TESA'
    else:
        target_arch = 'scalable MCM'
    print(f'START RL SEARCH, area constriant: 300 mm^2, Peak temp. constraint: 348K. Targe arch:{target_arch}\n')

    start_time = datetime.datetime.now()
    print(f'Start time:{start_time}')

    seed = 0
    num_cpu = 8 # Number of processes to use
    env = SubprocVecEnv([make_env(i, seed, sim_path_name, thermal_map, isRunBaseline1, isRunBaseline2, wkld_idpdt) for i in range(num_cpu)])
    
    ################# warmup ###################
    # if arg.warmup:
    #     n_warmup = 256
    #     for i in n_warmup:
    #         observation, reward, terminated, truncated, info = env.step(np.random.uniform(low=0.0, high=1.0, size=6))
    #         # convert to SB3 VecEnv api
    #         done = terminated or truncated
    #         if done:
    #             # save final observation where user can get it, then reset
    #             info["terminal_observation"] = observation
    #             observation, reset_info = env.reset()
                
    model = PPO('MlpPolicy', env, verbose=1, n_epochs = 5, normalize_advantage=True, ent_coef=0.1, vf_coef= 0.5, n_steps=256, batch_size=32, learning_rate=0.0003) # lr_default = 0.0003
    timesteps = 1000 * 25

    model.learn(total_timesteps=int(timesteps), progress_bar=True)

    # action_after_training, _ = model.predict(rand_obs, deterministic=True)
    # print(f'action after training:{action_after_training}, observation:{rand_obs}')
    print(f'leraning completed')

    current_time = str(datetime.datetime.now().month) + '_' + str(datetime.datetime.now().day) + '_' + str(datetime.datetime.now().hour) + '_' + str(datetime.datetime.now().minute)

    model_name = 'PPO_area_latency'+current_time
    model_path = os.path.join('.', model_name)
    model.save(model_path)

    for i in range(1, 10):
        rand_obs = env.observation_space.sample()
        action_after_training, _ = model.predict(rand_obs, deterministic=True)
        print(f'action after training:{action_after_training}, observation:{rand_obs}')

    end_time = datetime.datetime.now()
    print(f'Execution time:{end_time - start_time}')