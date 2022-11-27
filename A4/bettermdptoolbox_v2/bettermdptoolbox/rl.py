"""
Author: Miguel Morales
BSD 3-Clause License

Copyright (c) 2018, Miguel Morales
All rights reserved.
https://github.com/mimoralea/gdrl/blob/master/LICENSE
"""

"""
modified by: John Mansfield
"""

import numpy as np
from tqdm import tqdm
from bettermdptoolbox_v2.callbacks.callbacks import MyCallbacks
from bettermdptoolbox_v2.decorators.decorators import print_runtime
import gym


class RL:
    def __init__(self, env):
        self.env = env
        self.callbacks = MyCallbacks()
        self.render = False

    @staticmethod
    def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
        decay_steps = int(max_steps * decay_ratio)
        rem_steps = max_steps - decay_steps
        values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
        values = (values - values.min()) / (values.max() - values.min())
        values = (init_value - min_value) * values + min_value
        values = np.pad(values, (0, rem_steps), 'edge')
        return values


class QLearner(RL):
    def __init__(self, env):
        RL.__init__(self, env)

    @print_runtime
    def q_learning(self,
                   nS=None,
                   nA=None,
                   convert_state_obs=lambda state, done: state,
                   gamma=.99,
                   init_alpha=0.5,
                   min_alpha=0.01,
                   alpha_decay_ratio=0.5,
                   init_epsilon=1.0,
                   min_epsilon=0.1,
                   epsilon_decay_ratio=0.9,
                   n_episodes=10000,
                   random_reset=False,
                   random_action=False):
        '''
        random_reset: whether randomly reset env start state in early episodes
        '''
        #### modification ####
        # np.random.seed(0)
        ######################
        if nS is None:
            nS=self.env.observation_space.n
        if nA is None:
            nA=self.env.action_space.n
        pi_track = []
        Q = np.zeros((nS, nA), dtype=np.float64)+1e-3
        Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
        # select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
        #     if np.random.random() > epsilon \
        #     else np.random.randint(len(Q[state]))
        ##### modification #####
        if random_action:
            select_action = lambda state, Q, epsilon: np.random.choice(a=range(nA), p=Q[state]/Q[state].sum()) \
                if np.random.random() > epsilon \
                else np.random.randint(len(Q[state]))
            # select_action = lambda state, Q, epsilon: self.epsilon_greedy_exploration(state, Q, epsilon) if np.random.random() > epsilon \
            #     else np.random.randint(len(Q[state]))
        else:
            select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
                if np.random.random() > epsilon \
                else np.random.randint(len(Q[state]))
        ########################
        alphas = RL.decay_schedule(init_alpha,
                                min_alpha,
                                alpha_decay_ratio,
                                n_episodes)
        epsilons = RL.decay_schedule(init_epsilon,
                                  min_epsilon,
                                  epsilon_decay_ratio,
                                  n_episodes)
        for e in tqdm(range(n_episodes), leave=False):
            # if self.env.s>5:
            #     print('episode', e)
            self.callbacks.on_episode_begin(self)
            self.callbacks.on_episode(self, episode=e)
            # state, done = self.env.reset()[0], False
            #### modification ####
            if random_reset:
                state, done = self.reset_state(nS, epsilons[e])
            else:
                state, done = self.env.reset()[0], False
            # if self.env.s>5:
            #     print('start:', epsilons[e], state)
            ######################
            state = convert_state_obs(state, done)
            iter_cnt = 0
            while not done:
                if self.render:
                    self.env.render()
                # print(f'state {state}, self.s {self.env.s}')
                action = select_action(state, Q, epsilons[e])
                next_state, reward, done, _, prob = self.env.step(action)
                # print(f'next_state {next_state}, self.s {self.env.s}')
                # if state>10:
                # print(f'state {state}, action {action}, next_state {next_state}, reward {reward}, prob {prob}')
                self.callbacks.on_env_step(self)
                if iter_cnt > 1000:
                    break
                next_state = convert_state_obs(next_state,done)
                # td_target = reward + gamma * Q[next_state].max() * (not done)
                # td_error = td_target - Q[state][action]
                # Q[state][action] = Q[state][action] + alphas[e] * td_error
                ########## modification ########################
                td_target = reward + gamma * Q[next_state].max()
                td_error = td_target - Q[state][action]
                # if state>10:
                # print(f'td_target {td_target}, Q[state][action] {Q[state][action]}')
                Q[state][action] = Q[state][action] + alphas[e] * td_error
                # if state>5:
                # print(f'new Q[state][action] {Q[state][action]}')
                ################################################
                state = next_state
                iter_cnt += 1
            Q_track[e] = Q
            pi_track.append(np.argmax(Q, axis=1))
            self.render = False
            self.callbacks.on_episode_end(self)

        V = np.max(Q, axis=1)
        pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return Q, V, pi, Q_track, pi_track
    
    def reset_state(self, nS, epsilon):
        if np.random.random() < epsilon:
            if hasattr(self.env, 'total_step'):
                self.env.total_step = 0
            self.env.lastaction = None
            state = np.random.choice(range(nS))
            self.env.s = state
            return state, False
        else:
            return self.env.reset()[0], False
        
    def epsilon_greedy_exploration(self, state, Q, epsilon):
        if len(set(Q[state])) != 1:
            probs = np.ones(len(Q[state]), dtype=float) * epsilon / len(Q[state])
            best_action = np.argmax(Q[state])
            probs[best_action] += (1.0 - epsilon)
            return np.random.choice(a=range(len(Q[state])), p=probs)
        else:
            return np.random.choice(a=range(len(Q[state])))


class SARSA(RL):
    def __init__(self, env):
        RL.__init__(self, env)

    @print_runtime
    def sarsa(self,
              nS=None,
              nA=None,
              convert_state_obs=lambda state, done: state,
              gamma=.99,
              init_alpha=0.5,
              min_alpha=0.01,
              alpha_decay_ratio=0.5,
              init_epsilon=1.0,
              min_epsilon=0.1,
              epsilon_decay_ratio=0.9,
              n_episodes=10000):
        if nS is None:
            nS = self.env.observation_space.n
        if nA is None:
            nA = self.env.action_space.n
        pi_track = []
        Q = np.zeros((nS, nA), dtype=np.float64)
        Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
        select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
            if np.random.random() > epsilon \
            else np.random.randint(len(Q[state]))
        alphas = RL.decay_schedule(init_alpha,
                                min_alpha,
                                alpha_decay_ratio,
                                n_episodes)
        epsilons = RL.decay_schedule(init_epsilon,
                                  min_epsilon,
                                  epsilon_decay_ratio,
                                  n_episodes)

        for e in tqdm(range(n_episodes), leave=False):
            self.callbacks.on_episode_begin(self)
            self.callbacks.on_episode(self, episode=e)
            state, done = self.env.reset(), False
            state = convert_state_obs(state, done)
            action = select_action(state, Q, epsilons[e])
            while not done:
                if self.render:
                    self.env.render()
                next_state, reward, done, _ = self.env.step(action)
                self.callbacks.on_env_step(self)
                next_state = convert_state_obs(next_state, done)
                next_action = select_action(next_state, Q, epsilons[e])
                td_target = reward + gamma * Q[next_state][next_action] * (not done)
                td_error = td_target - Q[state][action]
                Q[state][action] = Q[state][action] + alphas[e] * td_error
                state, action = next_state, next_action
            Q_track[e] = Q
            pi_track.append(np.argmax(Q, axis=1))
            self.render = False
            self.callbacks.on_episode_end(self)

        V = np.max(Q, axis=1)
        pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return Q, V, pi, Q_track, pi_track
