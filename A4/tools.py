from hiive.mdptoolbox.example import forest
from hiive.mdptoolbox.util import check
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
import numbers
import matplotlib.pyplot as plt
from skimage.transform import resize
from forest import ForestEnv
from tqdm import tqdm
from copy import deepcopy

def convert_gym(env):
    '''
    Converts the transition probabilities provided by gymnasium envrionment to 
    MDPToolbox-compatible P and R arrays
    modified from https://github.com/hiive/hiivemdptoolbox/blob/master/hiive/mdptoolbox/openai.py
    
    gymnasium env.P format:
    {state1: {action1: [(prob_to_newstate1, newstate1, reward, terminated (boolean)),
                        (prob_to_newstate2, newstate2, reward, terminated (boolean)),
                        ...],
             action2: [(prob_to_newstate1, newstate1, reward, terminated (boolean)),
                        (prob_to_newstate2, newstate2, reward, terminated (boolean)),
                        ...],
             ...
            },
     state2: ... 
    }

    mdptoolbox P format: (A × S × S)
    mdptoolbox R format: (S × A)
    '''
    env.reset()
    transitions = env.P
    actions = int(re.findall(r'\d+', str(env.action_space))[0])
    states = int(re.findall(r'\d+', str(env.observation_space))[0])
    P = np.zeros((actions, states, states))
    R = np.zeros((states, actions))

    for state in range(states):
        for action in range(actions):
            for i in range(len(transitions[state][action])):
                tran_prob = transitions[state][action][i][0]
                state_ = transitions[state][action][i][1]
                R[state][action] += tran_prob*transitions[state][action][i][2]
                P[action, state, state_] += tran_prob
    return P, R


'''
######
Forest
######
transition probability (A × S × S) array P
           | p 1-p 0.......0  |
           | .  0 1-p 0....0  |
P[0,:,:] = | .  .  0  .       |
           | .  .        .    |
           | .  .         1-p |
           | p  0  0....0 1-p |

           | 1 0..........0 |
           | . .          . |
P[1,:,:] = | . .          . |
           | . .          . |
           | . .          . |
           | 1 0..........0 |
reward (S × A) matrix R
         |  0  |
         |  .  |
R[:,0] = |  .  |
         |  .  |
         |  0  |
         |  r1 |

         |  0  |
         |  1  |
R[:,1] = |  .  |
         |  .  |
         |  1  |
         |  r2 |
######
lake
######
https://gymnasium.farama.org/environments/toy_text/frozen_lake/
Frozen lake involves crossing a frozen lake from Start(S) to Goal(G) without falling into any Holes(H) by walking over the Frozen(F) lake. The agent may not always move in the intended direction due to the slippery nature of the frozen lake.
Action Space
The agent takes a 1-element vector for actions. The action space is (dir), where dir decides direction to move in which can be:

0: LEFT

1: DOWN

2: RIGHT

3: UP

Observation Space
The observation is a value representing the agent’s current position as current_row * nrows + current_col (where both the row and col start at 0). For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15. The number of possible observations is dependent on the size of the map. For example, the 4x4 map has 16 possible observations.

Rewards
Reward schedule:

Reach goal(G): +1

Reach hole(H): 0

Reach frozen(F): 0

Slippery world
Move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions.

For example, if action is left and is_slippery is True, then:

P(move left)=1/3
P(move up)=1/3
P(move down)=1/3
'''
# forest management
def get_forest(S=3, r1=4, r2=2, p=0.1):
    P, R = forest(S=S, r1=r1, p=p)
    n_state = P.shape[-1]
    R[(n_state-r2):n_state,1] = range(r2)
    return P, R

P_forest_small, R_forest_small = forest(S=5, r1=9, r2=5, p=0.1)
# P_forest_large, R_forest_large = forest(S=20, r1=9, r2=5, p=0.1)
# P_forest_small, R_forest_small = get_forest(S=5, r1=4, r2=5, p=0.2)
P_forest_large, R_forest_large = get_forest(S=20, r1=10, r2=15, p=0.15)
R_forest_large[:,1] = [0,1,1,1,1,1,3,6,10,15,21,28,36,45,55,66,78,91,105,120]
forest_small = ForestEnv(P_forest_small, R_forest_small, max_step=40)
forest_large = ForestEnv(P_forest_large, R_forest_large, max_step=40, simulate_R=False)
# print(forest_large.P[0])

# frozen lake
map2 = ['SH',
        'FG']
map4 = ['SFFH', 'FFFF', 'FFFF', 'FFHG']
map15 = ['SFFFFFHHHFFHFFF', 
         'FFFFFFFFFHFFFFF', 
         'HFFFFFFFFFFFHFH', 
         'FFFHFFFFFFFFFFF', 
         'HFFFFHFFFFFFFFF', 
         'FFFFFFFFFHFHFFF', 
         'FFHFFFFFFFFFFFF', 
         'FFFFFFFFFHFFFHF', 
         'HFFFFFFFFFFFFHH', 
         'FHFFFFFHHFFHHFF', 
         'FFFHFFFFFFFFFFF', 
         'HHHHFFHFHFHFFHF', 
         'FHFFFFFFFFFFFFF', 
         'FHFFFFFFFHHFFFF', 
         'FHFFFFFFHFFFFFG']
map20 = ['SHFFFFHFFFFFFHFFFFFF', 'FFHFFFFFFFFHFFFFFFFF', 'FFFHFFFFFFFFHFFFFFFF', 'FFFFFFFHFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFH', 'HFFFFFHFFFFFFFFFFFFF', 'FFHFFFFFFFFFHFFFFFFF', 'FFFFFFFFFFFFHFFFFFHF', 'FFFFFHFFFFFFFFFFFFFF', 'FFFFFFFFHHFFFFFHFFFF', 'FFHFFFFFFFFFHHFFFFFF', 'FFFFFFFFFFFFFFFFFHFF', 'FFFFFFFFFFFHFFFHFFFF', 'FFFFHFFFFFFFFFFFHFFF', 'FFFFFFHFFFHFFHFFFFFF', 'FFFFFFFFFHFFFFFFFFFF', 'FFFFFFHFFFFFFHFFFFFF', 'FFHFFFFFFFFFFFFFHFFF', 'FFFHFHFFFFFFFFFFHFFF', 'HHFFFFFHFFFFFHFFFFFG']
# lake_small = FrozenLakeEnv(desc=map4, is_slippery=True, render_mode="rgb_array")
lake_small = gym.make("FrozenLake-v1", desc=map4, is_slippery=True, render_mode="rgb_array")
P_lake_small, R_lake_small = convert_gym(lake_small)
# lake_large = FrozenLakeEnv(desc=map20, is_slippery=True, render_mode="rgb_array")
lake_large = gym.make("FrozenLake-v1", desc=map20, is_slippery=True, render_mode="rgb_array")
P_lake_large, R_lake_large = convert_gym(lake_large)
lake_large_for_train = deepcopy(lake_large)
for s in lake_large_for_train.P.keys():
    for a in lake_large_for_train.P[s].keys():
        for i, transition in enumerate(lake_large_for_train.P[s][a]):
            if (transition[3] is True) and (transition[1] != (len(lake_large_for_train.P)-1)):
                lake_large_for_train.P[s][a][i] = (transition[0], transition[1], -1000, transition[3])
            if transition[3] is False:
                lake_large_for_train.P[s][a][i] = (transition[0], transition[1], -1, transition[3])
            elif transition[1] == (len(lake_large_for_train.P)-1):
                lake_large_for_train.P[s][a][i] = (transition[0], transition[1], 1000, transition[3])

def visualize_lake(policy, P, lake):
    lake.reset()
    n_state = P.shape[-1]
    max_row = np.sqrt(n_state)
    axis0_mask = (lake.render() == np.zeros(shape=lake.render().shape)).sum(axis=(1,2)) != 512*3
    axis1_mask = (lake.render() == np.zeros(shape=lake.render().shape)).sum(axis=(0,2)) != 512*3
    actual_img = lake.render()[axis0_mask][:,axis1_mask,:]
    plt.imshow(actual_img)
    grid_width = (plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0])/max_row
    arrow_len = grid_width/3
    for s in list(range(n_state)):
        n_row_s = s // max_row
        n_col_s = s % max_row
        x_s = (n_col_s + 0.5) * grid_width
        y_s = (n_row_s + 0.5) * grid_width
        a = policy[s]
        if a == 0:
            x_s += arrow_len/2
            dx = -arrow_len
            dy = 0
        elif a == 1:
            y_s -= arrow_len/2
            dx = 0
            dy = arrow_len
        elif a == 2:
            x_s -= arrow_len/2
            dx = arrow_len
            dy = 0
        elif a == 3:
            y_s += arrow_len/2
            dx = 0
            dy = -arrow_len
        plt.arrow(x_s, y_s, dx=dx, dy=dy, width=grid_width/30, head_width=grid_width/6, color='g', alpha=0.5)
    if max_row > 10:
        ticks = np.arange(plt.gca().get_xlim()[0], plt.gca().get_xlim()[1]+1, grid_width*2)
        labels = np.arange(0, max_row+1, 2).astype(int)
    else:
        ticks = np.arange(plt.gca().get_xlim()[0], plt.gca().get_xlim()[1]+1, grid_width)
        labels = np.arange(0, max_row+1, 1).astype(int)
    plt.xticks(ticks, labels=labels)
    plt.yticks(ticks, labels=labels)
    plt.show()
    
def visualize_forest(policy):
    img0 = plt.imread('wait.png')
    img0 = resize(img0, (100,100))
    img1 = plt.imread('cut.png')
    img1 = resize(img1, (100,100))
    fig, axes = plt.subplots(1,len(policy), figsize=(1*len(policy), 1))
    for i, a in enumerate(policy):
        ax = axes[i]
        if a == 0:
            ax.imshow(img0)
        else:
            ax.imshow(img1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.title.set_text(f'{i}: wait' if a== 0 else f'{i}: cut')
    plt.show()

def eval_policy_forest(P, R, policy, episode=10, max_step=1000, start_s=0, verbose=True, plot_state_cnt=False, model=None, problem=None, xticks=None):
    state_cnt = [0]*len(policy)
    np.random.seed(0)
    if verbose:
        print(f'Test policy with steps = {max_step}')
    n_state = P.shape[-1]
    n_action = P.shape[0]
    reward_ls = []
    for i in range(episode):
        current_s = start_s
        state_cnt[current_s] += 1
        reward = 0
        step = 0
        while step < max_step:
            a = policy[current_s]
            next_s = np.random.choice(a=range(n_state), p=P[a, current_s])
            rwrd = R[current_s, a]
            if verbose == 2:
                print(f'Current state is {current_s}. Take action {a}. Next state is {next_s}. Reward is {rwrd}')
            current_s = next_s
            state_cnt[current_s] += 1
            reward += rwrd
            step += 1
        if verbose:
            print(f'Finish with {step} steps: reward={reward}')
        reward_ls.append(reward/max_step)
    if plot_state_cnt:
        state_time_pct = [round(x/sum(state_cnt),4)*100 for x in state_cnt]
        plt.figure()
        plt.bar(range(len(state_time_pct)), state_time_pct)
        if not xticks is None:
            plt.xticks(xticks, xticks)
        plt.xlabel('State')
        plt.ylabel('% of Time in State')
        plt.title(f'{model} - {problem}')
        plt.show()
    return round(np.mean(reward_ls), 6)

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution where each row specifies class probabilities.
    np_random: np.random.Generator
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return np.argmax(csprob_n > np_random.random())

def lake_step(a, lake, current_s):
    '''
    modified from https://github.com/Farama-Foundation/Gymnasium/blob/33e32178ffdf040a095bd2c856f6410bc7ec1306/gymnasium/envs/toy_text/frozen_lake.py
    a: action
    '''
    transitions = lake.P[current_s][a]
    i = categorical_sample([t[0] for t in transitions], np.random)
    p, s, r, t = transitions[i]
    return (int(s), r, t, False, {"prob": p})
    
def eval_policy_lake(lake, policy, episode=10, max_step=1000, verbose=True):
    '''
    verbose levels: [True, 2]
    '''
    np.random.seed(0)
    reward_ls = []
    step_ls = []
    for i in range(episode):
        current_s, _ = lake.reset()
        reward = 0
        step = 0
        while True:
            if step >= max_step:
                if verbose:
                    print(f'Exceed max_step of {max_step}. Stop episode.')
                break
            a = policy[current_s]
            next_s, rwrd, finish, _, _ = lake_step(a, lake, current_s)
            if verbose == 2:
                print(f'Current state is {current_s}. Take action {a}. Next state is {next_s}. Reward is {rwrd}')
            current_s = next_s
            reward += rwrd
            step += 1
            if finish:
                break
        if verbose:
            print(f'Finish with {step} steps: reward={reward}')
        reward_ls.append(reward)
        step_ls.append(step)
    if verbose:
        print(f'Finish with average steps of {np.mean(step_ls)}')
    return round(np.mean(reward_ls),6)

def tune_hyper(param_grid, algorithm, P, R, env=None, problem=None, verbose=True, **kwargs):
    '''
    select the hyper that has largest reward. If rewards are same, select the one with smallest iteration. If still the same, select parameter with the original input order (param order in pram_grid).
    **kwargs are policy args
    problem: 'forest' or 'lake'
    '''
    if (problem == 'lake') and (env is None):
        print('env must be provided.')
    policy_ls = []
    reward_ls = []
    iteration_ls = []
    params_ls = list(ParameterGrid(param_grid))
    for params in params_ls:
        alg = algorithm(P, R, **params)
        history = alg.run()
        policy = alg.policy
        policy_ls.append(policy)
        if hasattr(alg, 'iter'):
            iteration_ls.append(alg.iter)
        else:
            iteration_ls.append(alg.max_iter)
        if verbose:
            print(f'policy is {policy}')
        if problem == 'forest':
            reward_ls.append(eval_policy_forest(P, R, policy, verbose=False, **kwargs))
        elif problem == 'lake':
            reward_ls.append(eval_policy_lake(env, policy, verbose=False, **kwargs))
    result = pd.DataFrame(params_ls)
    result['reward'] = reward_ls
    result['policy'] = policy_ls
    result['iteration'] = iteration_ls
    best_idx = result.sort_values(by=['reward', 'iteration'], ascending=[False, True]).index[0]
    best_param = params_ls[best_idx]
    return best_param, result

def tune_hyper_QLearner(param_grid, algorithm, P, R, env=None, problem=None, verbose=True, param_repeat=1, **kwargs):
    '''
    select the hyper that has largest reward. If rewards are same, select the one with smallest iteration. If still the same, select parameter with the original input order (param order in pram_grid).
    **kwargs are eval_policy args
    problem: 'forest' or 'lake'
    '''
    if (problem == 'lake') and (env is None):
        print('env must be provided.')
    policy_ls = []
    reward_ls = []
    iteration_ls = []
    params_ls = list(ParameterGrid(param_grid)) 
    params_ls_repeat = params_ls* param_repeat
    for params in params_ls_repeat:
        env.reset()
        alg = algorithm(env)
        Q, V, _, Q_track, policy_track = alg.q_learning(**params)
        policy = policy_track[-1]
        policy_ls.append(policy)
        iteration_ls.append(len(policy_track))
        if verbose:
            print(f'Policy is {policy}. Params are {params}.')
        if problem == 'forest':
            reward_ls.append(eval_policy_forest(P, R, policy, verbose=False, **kwargs))
        elif problem == 'lake':
            reward_ls.append(eval_policy_lake(env, policy, verbose=False, **kwargs))
    result = pd.DataFrame(params_ls_repeat)
    result['reward'] = reward_ls
    # result['policy'] = policy_ls
    result['iteration'] = iteration_ls
    print(result.shape)
    result = result.groupby(list(param_grid.keys()))[['reward', 'iteration']].mean().reset_index()
    result = pd.DataFrame(params_ls).merge(result)
    best_idx = result.sort_values(by=['reward', 'iteration'], ascending=[False, True]).index[0]
    best_param = params_ls[best_idx]
    return best_param, result

def tune_hyper_QLearner_lake(param_grid, algorithm, P, R, env=None, env_train=None, problem=None, verbose=True, param_repeat=1, **kwargs):
    '''
    select the hyper that has largest reward. If rewards are same, select the one with smallest iteration. If still the same, select parameter with the original input order (param order in pram_grid).
    **kwargs are eval_policy args
    problem: 'forest' or 'lake'
    '''
    if (env_train is None) or (env is None):
        print('env and env_train must be provided.')
    policy_ls = []
    reward_ls = []
    iteration_ls = []
    params_ls = list(ParameterGrid(param_grid)) 
    params_ls_repeat = params_ls* param_repeat
    for params in params_ls_repeat:
        env.reset()
        alg = algorithm(env_train)
        Q, V, _, Q_track, policy_track = alg.q_learning(**params)
        policy = policy_track[-1]
        policy_ls.append(policy)
        iteration_ls.append(len(policy_track))
        if verbose:
            print(f'Policy is {policy}. Params are {params}.')
        if problem == 'forest':
            reward_ls.append(eval_policy_forest(P, R, policy, verbose=False, **kwargs))
        elif problem == 'lake':
            reward_ls.append(eval_policy_lake(env, policy, verbose=False, **kwargs))
    result = pd.DataFrame(params_ls_repeat)
    result['reward'] = reward_ls
    # result['policy'] = policy_ls
    result['iteration'] = iteration_ls
    print(result.shape)
    result = result.groupby(list(param_grid.keys()))[['reward', 'iteration']].mean().reset_index()
    result = pd.DataFrame(params_ls).merge(result)
    best_idx = result.sort_values(by=['reward', 'iteration'], ascending=[False, True]).index[0]
    best_param = params_ls[best_idx]
    return best_param, result

def tune_hyper_QLearning(param_grid, algorithm, P, R, env=None, problem=None, verbose=True, **kwargs):
    '''
    select the hyper that has largest reward. If rewards are same, select the one with smallest iteration. If still the same, select parameter with the original input order (param order in pram_grid).
    **kwargs are policy args
    problem: 'forest' or 'lake'
    '''
    if (problem == 'lake') and (env is None):
        print('env must be provided.')
    policy_ls = []
    reward_ls = []
    iteration_ls = []
    params_ls = list(ParameterGrid(param_grid))
    for params in tqdm(params_ls):
        alg = algorithm(P, R, **params)
        alg.Q = np.ones((alg.S, alg.A)) * 1e-3
        history = alg.run()
        policy = alg.policy
        policy_ls.append(policy)
        if hasattr(alg, 'iter'):
            iteration_ls.append(alg.iter)
        else:
            iteration_ls.append(alg.max_iter)
        if verbose:
            print(f'policy is {policy}')
        if problem == 'forest':
            reward_ls.append(eval_policy_forest(P, R, policy, verbose=False, **kwargs))
        elif problem == 'lake':
            reward_ls.append(eval_policy_lake(env, policy, verbose=False, **kwargs))
    result = pd.DataFrame(params_ls)
    result['reward'] = reward_ls
    result['policy'] = policy_ls
    result['iteration'] = iteration_ls
    best_idx = result.sort_values(by=['reward', 'iteration'], ascending=[False, True]).index[0]
    best_param = params_ls[best_idx]
    return best_param, result

def plot_hyper_curve(hyper, params, result, model_type, data_name, hyper_name=None, log_x=False, rot_x=False, ylim=None):
    if hyper_name is None:
        hyper_name = hyper
    result_hyper = result.copy()
    change_ticks = False
    for param in params:
        if param != hyper:
            result_hyper = result_hyper[result_hyper[param]==params[param]]
    hyper_list = result_hyper[hyper]
    reward_list = result_hyper['reward']
    if not isinstance(list(hyper_list)[0], numbers.Number):
        change_ticks = True
        tick_label_list = hyper_list
        hyper_list = list(range(len(hyper_list)))
    plt.figure(figsize=(5,3))
    plt.plot(hyper_list, reward_list, '-o')
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if log_x:
        plt.xscale('log')
    if change_ticks:
        rotation = 0
        if rot_x == True:
            rotation = 45
        plt.xticks(hyper_list, tick_label_list, rotation=rotation)
    plt.xlabel(f'{hyper_name}')
    plt.ylabel('Reward')
    plt.title(f'{model_type} Reward  - {data_name}')
    plt.show()

def plot_hyper_curve2(hyper, params, result, model_type, data_name, hyper_name=None, log_x=False, rot_x=False, ylim=None):
    change_ticks = False
    if hyper_name is None:
        hyper_name = hyper
    hyper_mean = result.groupby(hyper)['reward'].mean().reset_index()
    hyper_list = hyper_mean[hyper]
    reward_list = hyper_mean['reward']
    if not isinstance(list(hyper_list)[0], numbers.Number):
        change_ticks = True
        tick_label_list = hyper_list
        hyper_list = list(range(len(hyper_list)))
    plt.figure(figsize=(5,3))
    plt.plot(hyper_list, reward_list, '-o')
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if log_x:
        plt.xscale('log')
    if change_ticks:
        rotation = 0
        if rot_x == True:
            rotation = 45
        plt.xticks(hyper_list, tick_label_list, rotation=rotation)
    plt.xlabel(f'{hyper_name}')
    plt.ylabel('Reward')
    plt.title(f'{model_type} Reward  - {data_name}')
    plt.show()

def convergence_plot(model_type, data_name, history=None, alg=None, Q_track=None, policy_track=None, Q_diff_thresh=None, win=100):
    if model_type == 'Policy Iteration':
        if alg is None:
            print('alg must be provided.')
            return
        policy_diff_plot(alg, model_type, data_name)
        return
    elif model_type == 'Value Iteration':
        if history is None:
            print('history must be provided')
            return
        v_plot(history, model_type, data_name)
        v_diff_plot(history, model_type, data_name)
    elif model_type == 'Q-Learning':
        if (Q_track is None) or (policy_track is None):
            print('Q_track and policy_track must be provided')
            return 
        Q_plot(Q_track, policy_track, model_type, data_name, win=win)
        converge_idx = Q_diff_plot(Q_track, policy_track, model_type, data_name, thresh=Q_diff_thresh, win=win)
        return converge_idx
    
def policy_diff_plot(alg, model_type, data_name):
    policy_ls = np.array([x[1] for x in alg.p_cumulative])
    a = np.abs(np.diff(policy_ls, axis=0))
    policy_diff_ls = np.divide(a, a, out=np.zeros_like(a, dtype=float), where=a!=0).sum(axis=1)
    policy_diff_ls = np.append(policy_diff_ls, 0)
    iteration = range(2, len(policy_diff_ls)+2)
    plt.figure(figsize=(5,3))
    plt.plot(iteration, policy_diff_ls, '-o')
    plt.xlabel('Iteration')
    plt.ylabel('Delta (Policy Difference)')
    plt.title(f'{model_type} Convergence Plot  - {data_name}')
    plt.show()
    
def v_plot(history, model_type, data_name):
    mean_v = [x['Mean V'] for x in history]
    iteration = range(len(history))
    plt.figure(figsize=(5,3))
    plt.plot(iteration, mean_v, '-o')
    plt.xlabel('Iteration')
    plt.ylabel('Mean V')
    plt.title(f'{model_type} Convergence Plot  - {data_name}')
    plt.show()

def v_diff_plot(history, model_type, data_name):
    mean_v = np.array([x['Mean V'] for x in history])
    mean_v_diff = np.diff(mean_v)
    iteration = range(1, len(history))
    plt.figure(figsize=(5,3))
    plt.plot(iteration, mean_v_diff, '-o')
    plt.xlabel('Iteration')
    plt.ylabel('Delta (Mean V Difference)')
    plt.title(f'{model_type} Convergence Plot  - {data_name}')
    plt.show()
    
def Q_plot(Q_track, policy_track, model_type, data_name, win=100):
    mean_q = Q_track.max(axis=-1).mean(axis=-1)
    mean_q = np.convolve(mean_q, np.ones(win), 'valid') / win
    iteration = range(win-1, len(policy_track))
    plt.figure(figsize=(5,3))
    plt.plot(iteration, mean_q)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Q')
    plt.title(f'{model_type} Convergence Plot  - {data_name}')
    plt.show()

def Q_diff_plot(Q_track, policy_track, model_type, data_name, thresh=None, win=100):
    mean_q = Q_track.max(axis=-1).mean(axis=-1)
    mean_q = np.convolve(mean_q, np.ones(win), 'valid') / win
    mean_q_diff = np.absolute(np.diff(mean_q))
    iteration = range(win, len(policy_track))
    if thresh is None:
        thresh = mean_q_diff[-100:].mean()
    converge_idx = np.argmax(mean_q_diff<thresh)
    print(thresh, mean_q_diff[converge_idx])
    plt.figure(figsize=(5,3))
    plt.plot(iteration, mean_q_diff)
    plt.xlabel('Iteration')
    plt.ylabel('Delta (Mean Q Difference)')
    plt.title(f'{model_type} Convergence Plot  - {data_name}')
    plt.show()
    return converge_idx
    
def time_effect_plot(alg_small=None, alg_large=None, time_small=None, time_large=None, problem=None, model=None, iter_small=None, iter_large=None, n_state_small=None, n_state_large=None, iter_log=False, time_log=True):
    if iter_small is None:
        n_state_small = alg_small.P[0].shape[-1]
        n_state_large = alg_large.P[0].shape[-1]
        iter_small = alg_small.iter + 1
        iter_large = alg_large.iter + 1
    
    # time_ls_small = [x['Time'] for x in alg_small.run_stats]
    # time_small = np.sum(time_ls_small)
    # time_ls_large = [x['Time'] for x in alg_large.run_stats]
    # time_large = np.sum(time_ls_large)
    
    fig, ax = plt.subplots(1,2,figsize=(8,3))
    ax[0].bar([0,1], [iter_small, iter_large])
    ax[0].set_xticks([0,1], [f'small ({n_state_small})', f'large ({n_state_large})'])
    ax[0].text(0, iter_small, str(iter_small), ha='center')
    ax[0].text(1, iter_large, str(iter_large), ha='center')
    ax[0].set_ylabel('Iteration to Converge')
    if iter_log:
        ax[0].set_ylim(min(iter_small, iter_large)*0.1, max(iter_small, iter_large)*10)
        ax[0].set_yscale('log')
    else:
        ax[0].set_ylim(0, int(max(iter_small, iter_large)*1.1)+1)
    ax[1].bar([0,1], [time_small, time_large])
    ax[1].set_xticks([0,1], [f'small ({n_state_small})', f'large ({n_state_large})'])
    ax[1].text(0, time_small, str(round(time_small, 4)), ha='center')
    ax[1].text(1, time_large, str(round(time_large, 4)), ha='center')
    ax[1].set_ylabel('Time to Converge (Sec)')
    if time_log:
        ax[1].set_yscale('log')
        ax[1].set_ylim(round(min(time_small, time_large)*0.1, 6), round(max(time_small, time_large)*10, 4))
    else:
        ax[1].set_ylim(0, round(max(time_small, time_large)*1.1+1, 4))
    fig.suptitle(f'{model} Time Effect of State - {problem}')
    plt.tight_layout()
    plt.show()
    
def all_time_effect_plot(iter_list, time_list, labels, problem, logy=[False, True]):
    '''
    iter_list = [[iter_pi_small, iter_pi_large], [iter_vi_small, iter_vi_large], [iter_ql_small, iter_ql_large]]
    time_list = [[time_pi_small, time_pi_large], [time_vi_small, time_vi_large], [time_ql_small, time_ql_large]]
    '''
    
    iter_df = pd.DataFrame(index=['small', 'large'])
    time_df = pd.DataFrame(index=['small', 'large'])
    for i, label in enumerate(labels):
        iter_df[label] = iter_list[i]
    for i, label in enumerate(labels):
        time_df[label] = time_list[i]    
    
    # iter_df = pd.DataFrame({'PI': [iter_pi_small+1, iter_pi_large+1],
    #                         'VI': [iter_vi_small, iter_vi_large]},
    #                       index=['small', 'large'])
    # time_df = pd.DataFrame({'PI': [time_pi_small, time_pi_large],
    #                         'VI': [time_vi_small, time_vi_large]},
    #                       index=['small', 'large']).round(4)
    
    fig, ax = plt.subplots(1,2,figsize=(8,3))
    axes0 = iter_df.plot.bar(rot=0, logy=logy[0], ax=ax[0])
    axes1 = time_df.plot.bar(rot=0, logy=logy[1], ax=ax[1])
    for patch in axes0.patches:
        bl = patch.get_xy()
        x = 0.5 * patch.get_width() + bl[0]
        # change 0.92 to move the text up and down
        y = patch.get_height() + bl[1] 
        axes0.text(x,y,str(patch.get_height()), ha='center')
    for patch in axes1.patches:
        bl = patch.get_xy()
        x = 0.5 * patch.get_width() + bl[0]
        # change 0.92 to move the text up and down
        y = patch.get_height() + bl[1] 
        axes1.text(x,y,str(patch.get_height()), ha='center')
    if logy[0]:
        ax[0].set_ylim(round(iter_df.min().min()*0.1, 6), round(iter_df.max().max()*100, 4))
    else:
        ax[0].set_ylim(0, int(iter_df.max().max()*1.1)+1)
    ax[0].set_ylabel('Iteration to Converge')
    if logy[1]:
        ax[1].set_ylim(round(time_df.min().min()*0.1, 6), round(time_df.max().max()*100, 4))
    else:
        ax[0].set_ylim(0, int(time_df.max().max()*1.1)+1)
    ax[1].set_ylabel('Time to Converge (Sec)')
    ax[0].legend(loc='upper left')
    ax[1].legend(loc='upper left')
    fig.suptitle(f'Time Effect of State - {problem}')
    plt.tight_layout()
    plt.show()
###