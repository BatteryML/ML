import numpy as np
from tqdm import tqdm

def samplefrom(distribution):
    return (np.random.choice(len(distribution), 1, p=distribution))[0]

def playtransition(P, R, state, action):
        nextstate = samplefrom(P[action][state])
        return nextstate, R[state][action]

def epsilon_greedy_exploration(Q, epsilon, num_actions):
    def policy_exp(state):
        probs = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        probs[best_action] += (1.0 - epsilon)
        return probs
    return policy_exp

def q_learning(P, R, num_episodes, T_max, epsilon=0.01, gamma=0.99):
    nS = P.shape[-1]
    nA = P.shape[0]
    Q = np.zeros((nS, nA))
    episode_rewards = np.zeros(num_episodes)
    policy = np.ones(nS)
    V = np.zeros((num_episodes, nS))
    N = np.zeros((nS, nA))
    for i_episode in tqdm(range(num_episodes)): 
        # epsilon greedy exploration
        greedy_probs = epsilon_greedy_exploration(Q, epsilon, nA)
        state = np.random.choice(np.arange(nS))
        for t in range(T_max):
            # epsilon greedy exploration
            action_probs = greedy_probs(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward = playtransition(P, R, state, action)
            episode_rewards[i_episode] += reward
            N[state, action] += 1
            alpha = 1/(t+1)**0.8
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + gamma * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
            state = next_state
        V[i_episode,:] = Q.max(axis=1)
        policy = Q.argmax(axis=1)
        
    return V, policy, episode_rewards, N, Q