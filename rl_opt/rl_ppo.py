import argparse
import os, sys

from gym import Env
from gym import spaces
import numpy as np
import random
import datetime
from stable_baselines3 import PPO, A2C, SAC, TD3, DQN, HerReplayBuffer

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
    ap.add_argument('-hp', '--hotspot_path', type=str, default='../../HotSpot',
                    help='the hotspot path')
    ap.add_argument('-sp', '--sim_path', type=str, default='../tmp',
                    help='thermal simulation path')
    ap.add_argument('-wi', '--wkld_idpdt', type=int, default=0,
                    help='peak temp. is max(peak_temp) ')
    ap.add_argument('-maxT', '--max_temp', type=int, default=348,
                help='peak temperature constraints ')
    ap.add_argument('-maxA', '--max_area', type=int, default=300,
            help='max area constraints (mm^2)')
    return ap

arg = argparser().parse_args()
isSearchMapping = arg.map
hotspot_path = arg.hotspot_path
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
print(f'START SCBO TWO STAGE SEARCH, area constriant: 300 mm^2, Peak temp. constraint: 350K. Targe arch:{target_arch}\n')

start_time = datetime.datetime.now()
print(f'Start time:{start_time}')

def action_refined(action):
    cs = action[0] + 1 # cores in totlal, start from 1
    cc = action[1] + 1 if action[1] + 1 < cs else cs # chiplet cut 
    # ci = action[2] # chiplet interval space
    hsa = action[2] + 1 # height of systolic array / matrix unit
    wsa = action[3] + 1# width of systolic array / matrix unit
    ubuf = action[4] + 1
    nop_bw = action[5] + 1
    action_new = [cs, cc, hsa, wsa, ubuf, nop_bw]
    return action_new

def param_regulator(x):
    xx = x[0] # cores in x dim
    yy = x[1] # cores in y dim
    cx = xx
    cy = yy
    ci = 2 # chiplet interval space
    hsa = x[2] # height of systolic array / matrix unit
    wsa = x[3] # width of systolic array / matrix unit
    ubf = x[4] # uniformed buffer size
    nop = x[5]
    dram = 4
    xx, yy = int(xx), int(yy)
    cx, cy = int(cx), int(cy) 
    chipletIntvl = 0.0004 + int(ci) * 0.0003
    h_sa = 16 * int(hsa)  
    w_sa = 16 * int(wsa)
    ubuf_size = 128 * 2** int(ubf) *1024
    nop_bw = 16 * int(nop)
    dram_bw = 32 * int(dram)
    sys_info = [xx, yy,cx, cy, chipletIntvl, h_sa, w_sa, ubuf_size, nop_bw, dram_bw]
    return sys_info

class CustomEnv(Env):
    def __init__(self, sim_path, thermal_map, maxA, maxT, isRunBaseline1, isRunBaseline2, wkld_idpdt):
        self.max_interposer_area = maxA      # unit m^2
        self.max_temperature = maxT # unit K
        self.action_space = spaces.MultiDiscrete([8,8,16,16,5,8])
        self.obs_space_low = np.array([0, 0, -1e9])
        self.obs_space_high = np.array([self.max_interposer_area, self.max_temperature, -1])
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
        self.current_obs = np.array(
            [self.max_interposer_area, self.max_temperature, -1])
        return self.current_obs

    def step(self, action):
        """Takes action and returns next observation, reward done and optionally additional info"""
        done = False
        # print(f'action before:{action}')
        action = action_refined(action)
        # print(f'action after:{action}')
        sys_info = param_regulator(action)
        # evaluator = chiplet_evaluator(nn_wld, interposer_cons=0.03, hotspot_path='../../HotSpot', sim_path='../tmp', sys_info= sys_info)
        # evaluator.generate_hardware()
        evaluator = chiplet_evaluator( hotspot_path=hotspot_path, sim_path=self.sim_path, sys_info= sys_info, thermal_map=self.thermal_map,baseline1=self.isRunBaseline1, baseline2=self.isRunBaseline2,wkld_idpdt=self.wkld_idpdt)
        evaluator.generate_hardware()
        delay, energy, die_yield=evaluator.evaluate()
        EDPY_cost = - energy * delay / die_yield
        peak_temp = evaluator.evaluate_thermal()
        area = evaluator.evaluate_area()
        print(f'sys_info:{sys_info}, Area: {area}, peak temp: {peak_temp}')
        # print(f'Area of Computing die: {compute_die_area}, IO die: {IO_die_area}')
        print(f'E: {energy} D: {delay}, Yield: {die_yield}, total area: {area}. The EDYP cost is {EDPY_cost}')
        self.current_obs[0] = area
        self.current_obs[1] = peak_temp
        self.current_obs[2] = EDPY_cost

        next_obs = self.current_obs
        reward = EDPY_cost - 100 * (max(0, (peak_temp - self.max_temperature))) - 1e4 * max(0, area - self.max_interposer_area)
        # if peak_temp > self.max_temperature or area > self.max_interposer_area:
        #     reward +=  EDPY_cost - (peak_temp - self.max_temperature)
        # else:
        #     reward = EDPY_cost

        self.time_step += 1
        if self.time_step > 1:
            done = True
        self.current_obs = next_obs

        return self.current_obs, reward, done, {}

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass


# gym.envs.register(
#     id = 'ChipletEnv',
#     entry_point = 'gym.envs.classic_control:CustomEnv',
#     max_episode_steps = 10,
# )

# env = gym.make('ChipletEnv')
env = CustomEnv(sim_path, thermal_map,max_area, max_temp,  isRunBaseline1, isRunBaseline2, wkld_idpdt)
env.reset()
env.step(env.action_space.sample())
rand_obs = env.observation_space.sample()
model = PPO('MlpPolicy', env, verbose=1, n_epochs = 5, normalize_advantage=True, ent_coef=0.1, vf_coef= 0.5, n_steps=256, batch_size=32, learning_rate=0.0003) # lr_default = 0.0003
action_before_training, _ = model.predict(rand_obs, deterministic=False)
print(f'action_before_training:{action_before_training}, Observation:{rand_obs}')
timesteps = 1024 *20

model.learn(total_timesteps=int(timesteps), progress_bar=True)

action_after_training, _ = model.predict(rand_obs, deterministic=True)
print(f'action after training:{action_after_training}, observation:{rand_obs}')
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