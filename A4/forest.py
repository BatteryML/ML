# from contextlib import closing
# from io import StringIO
# from os import path
# from typing import List, Optional

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from copy import deepcopy
# from gymnasium.error import DependencyNotInstalled

class ForestEnv(Env):
    """
    convert mdptoolbox P,R to gymnasium env
    """

    def __init__(
        self,
        P,
        R,
        simulate_R=False,
        max_step=40
    ):
        nA = P.shape[0]
        nS = P.shape[-1]
        '''
        gymnasium env.P format:
        {state1: {action1: [(prob_to_newstate1, newstate1, reward, terminated (boolean)),
                            (prob_to_newstate2, newstate2, reward, terminated (boolean)),
                            ...],
                 action2: [(prob_to_newstate1, newstate1, reward, terminated (boolean)),
                            (prob_to_newstate2, newstate2, reward, terminated (boolean)),
                            ...],
                 ...
                }
        }
        '''
        self.max_step = max_step
        self.reset()
        
        if simulate_R:
            R = self.simu_R(P, R)
            # print(R)
            self.P = {int(s): {int(a): [(prob, s_next, R[int(s_next), int(a)], False) for s_next,prob in enumerate(P[int(a),int(s)])] for a in range(nA)} for s in range(nS)}
        else:
            self.P = {int(s): {int(a): [(prob, s_next, R[int(s), int(a)], False) for s_next,prob in enumerate(P[int(a),int(s)])] for a in range(nA)} for s in range(nS)}

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

    def step(self, a):
        self.total_step += 1
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        # print(f'inside step: current state {self.s}, action {a}, next_state {int(s)}, reward {r}, prob {p}')
        self.s = s
        self.lastaction = a
        # print(f'inside step: current state {self.s}')
        # print(self.total_step)
        if self.total_step >= self.max_step:
            return (int(s), r, True, False, {"prob": p})
        else:
            return (int(s), r, False, False, {"prob": p})

    def reset(self):
        self.s = 0
        self.lastaction = None
        self.total_step = 0

        return int(self.s), {"prob": 1}
    
    def simu_R(self, P, R):
        episode = 100000
        max_step = self.max_step
        np.random.seed(0)
        n_state = P.shape[-1]
        start_s = np.random.choice(range(n_state))
        n_action = P.shape[0]
        state_ls = []
        reward_ls = []
        for i in range(episode):
            current_s = start_s
            reward = 0
            step = 0
            while step < max_step:
                state_ls.append(current_s)
                a = np.random.choice(range(n_action))
                next_s = np.random.choice(a=range(n_state), p=P[a, current_s])
                rwrd = R[current_s, a]
                current_s = next_s
                reward_ls.append(rwrd)
                step += 1
        r_df = pd.DataFrame({'state': state_ls, 'reward': reward_ls})
        r_df = (r_df.groupby('state').sum()/episode).reindex(range(n_state)).fillna(0)
        R = deepcopy(R)
        R[:,1] = r_df['reward']
        return R



# Elf and stool from https://franuka.itch.io/rpg-snow-tileset
# All other assets by Mel Tillery http://www.cyaneus.com/