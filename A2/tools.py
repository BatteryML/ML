import mlrose_hiive as mlrose
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np

def rhcRunner(problem, iteration_list, max_attempts, restart_num, seed=0, plot=True, return_curve=False):
    '''
    randomized hill climbing
    1. neighbor means two digits in the states are exchanged.
    2. Indicated in source code, max_attempts are maximum attempts for continuous not finding a better neighbor in each restart. When a better neighbor is found, attempt resets to 0, otherwise attemps are accumulable across interations.
    
    attempts = 0
    iters = 0

    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1

        # Find random neighbor and evaluate fitness
        next_state = problem.random_neighbor()
        next_fitness = problem.eval_fitness(next_state)

        # If best neighbor is an improvement,
        # move to that state and reset attempts counter
        if next_fitness > problem.get_fitness():
            problem.set_state(next_state)
            attempts = 0

        else:
            attempts += 1

        if curve:
            fitness_curve.append(problem.get_fitness())
    '''
    rhc = mlrose.RHCRunner(problem=problem,
                experiment_name='rhc',
                seed=seed,
                iteration_list=iteration_list, # last element is max_iter?
                max_attempts=max_attempts,
                restart_list=[restart_num])   
    _, df_run_curves = rhc.run()
    best_fitness = df_run_curves['Fitness'].max()
    best_param = df_run_curves[df_run_curves['Fitness']==best_fitness][['current_restart', 'max_iters', 'Iteration']].to_dict('records')[0]
    f_evals = df_run_curves[df_run_curves['Fitness']==best_fitness]['FEvals'].iloc[0]
    iter_num = best_param['Iteration']
    time_s = df_run_curves.groupby('current_restart')['Time'].max().sum() - df_run_curves['Time'].loc[0]
    iter_time_s = time_s/(len(df_run_curves)-1)
    converge_time_s = iter_time_s * (df_run_curves[df_run_curves['current_restart']<best_param['current_restart']].shape[0] + best_param['Iteration'])
    # plot best restart
    if plot:
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        df_run_curves[df_run_curves['current_restart']==best_param['current_restart']].plot(
            x='Iteration', y='Fitness', style='--o', ax=ax[0])
        df_run_curves.groupby('current_restart')['Fitness'].max().plot(label='Fitness', 
                                                                       style='--o', ax=ax[1])
        plt.legend()
        plt.show()
    if return_curve:
        best_df_run_curves = df_run_curves[df_run_curves['current_restart']==best_param['current_restart']]
        return int(best_fitness), best_param, int(iter_num), round(converge_time_s, 6), round(iter_time_s, 6), int(f_evals), best_df_run_curves
    else:
        return int(best_fitness), best_param, int(iter_num), round(converge_time_s, 6), round(iter_time_s, 6), int(f_evals)

def saRunner(problem, iteration_list, max_attempts, temperature_list, decay_list=[mlrose.GeomDecay], seed=0, plot=True, return_curve=False):
    '''
    simulated annealing
    1. worse solutions could be accepted with probabilities
    2. chance to get out of local optima and reach global optima
    3. probabilites are proportional to temperature, which gradually decreases
    4. Similar to RHC, max_attempts are maximum attempts for continuous not finding a better neighbor in each restart. When a better neighbor is found, attempt resets to 0, otherwise attemps are accumulable across interations.
    '''
    sa = mlrose.SARunner(problem=problem,
                  experiment_name='sa',
                  seed=seed,
                  iteration_list=iteration_list,
                  max_attempts=max_attempts,
                  temperature_list=temperature_list,
                  decay_list=decay_list)
    # the two data frames will contain the results
    _, df_run_curves = sa.run() 
    df_run_curves['Temperature'] = df_run_curves['Temperature'].astype(str).astype(int)
    best_fitness = df_run_curves['Fitness'].max()
    best_param = df_run_curves[df_run_curves['Fitness']==best_fitness][['Temperature', 'max_iters', 'Iteration']].to_dict('records')[0]
    f_evals = df_run_curves[df_run_curves['Fitness']==best_fitness]['FEvals'].iloc[0]
    iter_num = best_param['Iteration']
    time_s = df_run_curves.groupby('Temperature')['Time'].max().sum() - df_run_curves['Time'].loc[0]
    iter_time_s = time_s/(len(df_run_curves)-1)
    converge_time_s = iter_time_s * best_param['Iteration']
    # plot best 
    if plot:
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        df_run_curves[df_run_curves['Temperature']==best_param['Temperature']].plot(
            x='Iteration', y='Fitness', style='--o', ax=ax[0])
        df_run_curves.groupby('Temperature')['Fitness'].max().plot(label='Fitness', 
                                                                       style='--o', logx=True, ax=ax[1])
        plt.legend()
        plt.show()
    if return_curve:
        best_df_run_curves = df_run_curves[df_run_curves['Temperature']==best_param['Temperature']]
        return int(best_fitness), best_param, int(iter_num), round(converge_time_s, 6), round(iter_time_s, 6), int(f_evals), best_df_run_curves
    else:
        return int(best_fitness), best_param, int(iter_num), round(converge_time_s, 6), round(iter_time_s, 6), int(f_evals)

def gaRunner(problem, iteration_list, max_attempts, population_sizes, mutation_rates, seed=0, plot=True, return_curve=False):
    '''
    genetic algorithm
    1. select initial population
    2. parents crossover
    3. mutation
    4. remove least fitness solutions
    5. states contain binary digits, mutation means flipping 0 and 1
    6. for problems whose states contains other integers, 
    permutation is used instead of mutation
    7. population keeps the same size.
    '''
    ga = mlrose.GARunner(problem=problem,
                experiment_name='ga',
                  seed=seed,
                  iteration_list=iteration_list,
                  max_attempts=max_attempts,
                  population_sizes=population_sizes,
                  mutation_rates=mutation_rates)

    # the two data frames will contain the results
    _, df_run_curves = ga.run()   
    best_fitness = df_run_curves['Fitness'].max()
    best_param = df_run_curves[df_run_curves['Fitness']==best_fitness][['Population Size', 'Mutation Rate', 'max_iters', 'Iteration']].to_dict('records')[0]
    f_evals = df_run_curves[df_run_curves['Fitness']==best_fitness]['FEvals'].iloc[0]
    iter_num = best_param['Iteration']
    time_s = df_run_curves.groupby(['Population Size', 'Mutation Rate'])['Time'].max().sum() - df_run_curves['Time'].loc[0]
    iter_time_s = time_s/(len(df_run_curves)-1)
    converge_time_s = iter_time_s * best_param['Iteration']
    if plot:
        fig, ax = plt.subplots(2,2, figsize=(12,8))
        df_run_curves[(df_run_curves['Population Size']==best_param['Population Size'])&\
                     (df_run_curves['Mutation Rate']==best_param['Mutation Rate'])].plot(
            x='Iteration', y='Fitness', style='--o', ax=ax[0,0])
        df_run_curves[df_run_curves['Mutation Rate']==best_param['Mutation Rate']].groupby(
            'Population Size')['Fitness'].max().plot(label='Fitness', style='--o', logx=True, ax=ax[1,0])
        ax[1,0].legend()
        df_run_curves[df_run_curves['Population Size']==best_param['Population Size']].groupby(
            'Mutation Rate')['Fitness'].max().plot(label='Fitness', style='--o', logx=True, ax=ax[1,1])
        ax[1,1].legend()
        plt.show()
    if return_curve:
        best_df_run_curves = df_run_curves[(df_run_curves['Population Size']==best_param['Population Size'])&(df_run_curves['Mutation Rate']==best_param['Mutation Rate'])]
        return int(best_fitness), best_param, int(iter_num), round(converge_time_s, 6), round(iter_time_s, 6), int(f_evals), best_df_run_curves
    else:
        return int(best_fitness), best_param, int(iter_num), round(converge_time_s, 6), round(iter_time_s, 6), int(f_evals)

def mimicRunner(problem, iteration_list, max_attempts, population_sizes, keep_percent_list, seed=0, plot=True, return_curve=False):
    mmc = mlrose.MIMICRunner(problem=problem,
                      experiment_name='mimic',
                      seed=seed,
                      iteration_list=iteration_list,
                      max_attempts=max_attempts,
                      population_sizes=population_sizes,
                      keep_percent_list=keep_percent_list,
                      use_fast_mimic=True)

    # the two data frames will contain the results
    _, df_run_curves = mmc.run()
    best_fitness = df_run_curves['Fitness'].max()
    best_param = df_run_curves[df_run_curves['Fitness']==best_fitness][['Population Size', 'Keep Percent', 'max_iters', 'Iteration']].to_dict('records')[0]
    f_evals = df_run_curves[df_run_curves['Fitness']==best_fitness]['FEvals'].iloc[0]
    iter_num = best_param['Iteration']
    time_s = df_run_curves.groupby(['Population Size', 'Keep Percent'])['Time'].max().sum() - df_run_curves['Time'].loc[0]
    iter_time_s = time_s/(len(df_run_curves)-1)
    converge_time_s = iter_time_s * best_param['Iteration']
    if plot:
        fig, ax = plt.subplots(2,2, figsize=(12,8))
        df_run_curves[(df_run_curves['Population Size']==best_param['Population Size'])&\
                     (df_run_curves['Keep Percent']==best_param['Keep Percent'])].plot(
            x='Iteration', y='Fitness', style='--o', ax=ax[0,0])
        df_run_curves[df_run_curves['Keep Percent']==best_param['Keep Percent']].groupby(
            'Population Size')['Fitness'].max().plot(label='Fitness', style='--o', logx=True, ax=ax[1,0])
        df_run_curves[df_run_curves['Population Size']==best_param['Population Size']].groupby(
            'Keep Percent')['Fitness'].max().plot(label='Fitness', style='--o', ax=ax[1,1])
        plt.legend()
        plt.show()
    if return_curve:
        best_df_run_curves = df_run_curves[(df_run_curves['Population Size']==best_param['Population Size'])&(df_run_curves['Keep Percent']==best_param['Keep Percent'])]
        return int(best_fitness), best_param, int(iter_num), round(converge_time_s, 6), round(iter_time_s, 6), int(f_evals), best_df_run_curves
    else:
        return int(best_fitness), best_param, int(iter_num), round(converge_time_s, 6), round(iter_time_s, 6), int(f_evals)

def repeat_algorithm(algorithm, params, repeat=5):
    best_fitness_ls = []
    iter_num_ls = []
    converge_time_s_ls = []
    iter_time_s_ls = []
    f_evals_ls = []
    for i in range(repeat):
        best_fitness, best_param, iter_num, converge_time_s, iter_time_s, f_evals= algorithm(seed=i, plot=False, **params)
        best_fitness_ls.append(best_fitness)
        iter_num_ls.append(iter_num)
        converge_time_s_ls.append(converge_time_s)
        iter_time_s_ls.append(iter_time_s)
        f_evals_ls.append(f_evals)
    result = pd.DataFrame({
        'best_fitness': [max(best_fitness_ls)],
        'mean_fitness': [int(np.mean(best_fitness_ls))],
        'iter_num': [int(np.mean(iter_num_ls))],
        'converge_time_s': [round(np.mean(converge_time_s_ls), 6)],
        'iter_time': [round(np.mean(iter_time_s_ls), 6)],
        'func_evals': [int(np.mean(f_evals_ls))]
    })
    return result

def compare(problem, RHC_params, SA_params, GA_params, MIMIC_params):
    RHC_params['problem'] = problem
    SA_params['problem'] = problem
    GA_params['problem'] = problem
    MIMIC_params['problem'] = problem
    rhc_result = repeat_algorithm(rhcRunner, RHC_params)
    rhc_result.index = ['RHC']
    sa_result = repeat_algorithm(saRunner, SA_params)
    sa_result.index = ['SA']
    ga_result = repeat_algorithm(gaRunner, GA_params)
    ga_result.index = ['GA']
    mimic_result = repeat_algorithm(mimicRunner, MIMIC_params)
    mimic_result.index = ['MIMIC']
    result = pd.concat([rhc_result, sa_result, ga_result, mimic_result])
    return result

def tune_max_attempts_plot(algorithm, algorithm_name, attempts_list, best_params, figsize=(3,2)):
    fitness_list = []
    best_params = deepcopy(best_params)
    del best_params['max_attempts']
    for max_attempts in attempts_list:
        best_fitness, best_param, iter_num, time_s, iter_time_s, f_evals = algorithm(max_attempts=max_attempts,plot=False, **best_params)
        fitness_list.append(best_fitness)
    plt.figure(figsize=figsize)
    plt.plot(attempts_list, fitness_list, '--o')
    plt.xlabel('max_attempts')
    plt.ylabel('Best Fitness')
    plt.title(f'Tune max_attempts -- {algorithm_name}')
    plt.show()
    return fitness_list

def problem_size(algorithm, problem_set, best_params, repeat=1):
    fitness_list = []
    time_list = []
    f_evals_list = []
    params = deepcopy(best_params)
    for problem in problem_set:
        params['problem'] = problem
        result = repeat_algorithm(algorithm, params, repeat=repeat)
        fitness_list.append(result['mean_fitness'].iloc[0])
        time_list.append(result['converge_time_s'].iloc[0])
        f_evals_list.append(result['func_evals'].iloc[0])
    return fitness_list, time_list, f_evals_list

def problem_size_plot(N_list, problem_set, best_params_list, repeat=1, figsize=(10,6)):
    result = []
    algorithm_list = [rhcRunner, saRunner, gaRunner, mimicRunner]
    algorithm_name_list = ['RHC', 'SA', 'GA', 'MIMIC']
    fig, ax = plt.subplots(2,2, figsize=figsize)
    for algorithm, algorithm_name, best_params in zip(algorithm_list, algorithm_name_list, best_params_list):
        print(f'plot {algorithm_name}...')
        fitness_list, time_list, f_evals_list = problem_size(algorithm, problem_set, best_params, repeat=repeat)
        sub = pd.DataFrame({
            f'fitness_{algorithm_name}': fitness_list,
            f'converge_time_{algorithm_name}': time_list,
            f'func_evals_{algorithm_name}': f_evals_list
        })
        result.append(sub)   
        ax[0,0].plot(N_list, fitness_list, '--o', label=algorithm_name)
        ax[0,1].plot(N_list, time_list, '--o', label=algorithm_name)
        ax[1,0].plot(N_list, f_evals_list, '--o', label=algorithm_name)
    ax[0,0].set_xlabel('Problem Size')
    ax[0,0].set_ylabel('Best Fitness')
    ax[0,0].title.set_text(f'Fitness on Problem Size')
    ax[0,0].legend()
    ax[0,1].set_yscale('log')
    ax[0,1].set_xlabel('Problem Size')
    ax[0,1].set_ylabel('Coverge Time (s)')
    ax[0,1].title.set_text(f'Time on Problem Size')
    ax[0,1].legend()
    ax[1,0].set_yscale('log')
    ax[1,0].set_xlabel('Problem Size')
    ax[1,0].set_ylabel('Function Evaluations')
    ax[1,0].title.set_text(f'Function Evalustions on Problem Size')
    ax[1,0].legend()
    plt.tight_layout()
    plt.show()  
    result = pd.concat(result, axis=1).T
    result.columns = N_list
    return result
    
def iteration(algorithm, iter_list, best_params, repeat=1):
    fitness_list = []
    time_list = []
    f_evals_list = []
    params = deepcopy(best_params)
    for max_iter in iter_list:
        params['iteration_list'] = [max_iter]
        result = repeat_algorithm(algorithm, params, repeat=repeat)
        fitness_list.append(result['mean_fitness'].iloc[0])
        time_list.append(result['converge_time_s'].iloc[0])
        f_evals_list.append(result['func_evals'].iloc[0])
    return fitness_list, time_list, f_evals_list

def iteration_plot(iter_list, best_params_list, repeat=1, figsize=(10,6)):
    result = []
    algorithm_list = [rhcRunner, saRunner, gaRunner, mimicRunner]
    algorithm_name_list = ['RHC', 'SA', 'GA', 'MIMIC']
    fig, ax = plt.subplots(2,2, figsize=figsize)
    for algorithm, algorithm_name, best_params in zip(algorithm_list, algorithm_name_list, best_params_list):
        print(f'plot {algorithm_name}...')
        fitness_list, time_list, f_evals_list = iteration(algorithm, iter_list, best_params, repeat=repeat)
        sub = pd.DataFrame({
            f'fitness_{algorithm_name}': fitness_list,
            f'converge_time_{algorithm_name}': time_list,
            f'func_evals_{algorithm_name}': f_evals_list
        })
        result.append(sub)    
        ax[0,0].plot(iter_list, fitness_list, '--o', label=algorithm_name)
        ax[0,1].plot(iter_list, time_list, '--o', label=algorithm_name)
        ax[1,0].plot(iter_list, f_evals_list, '--o', label=algorithm_name)
    ax[0,0].set_xlabel('Max Iteration')
    ax[0,0].set_ylabel('Best Fitness')
    ax[0,0].title.set_text(f'Fitness on Max Iteration')
    ax[0,0].legend()
    ax[0,1].set_yscale('log')
    ax[0,1].set_xlabel('Max Iteration')
    ax[0,1].set_ylabel('Coverge Time (s)')
    ax[0,1].title.set_text(f'Time on Max Iteration')
    ax[0,1].legend()
    ax[1,0].set_yscale('log')
    ax[1,0].set_xlabel('Max Iteration')
    ax[1,0].set_ylabel('Function Evaluations')
    ax[1,0].title.set_text(f'Function Evalustions on Max Iteration')
    ax[1,0].legend()
    plt.tight_layout()
    plt.show()
    result = pd.concat(result, axis=1).T
    result.columns = iter_list
    return result

def fitness_per_iteration(algorithm, best_params):
    best_fitness, best_param, iter_num, time_s, iter_time_s, f_evals, df_run_curves = algorithm(return_curve=True, plot=False, **best_params)
    return df_run_curves[['Iteration', 'Fitness', 'Time', 'FEvals']]
    
def fitness_per_iteration_plot(best_params_list, figsize=(10,6)):
    curve_list = []
    longest = 0
    iter_s = pd.Series()
    algorithm_list = [rhcRunner, saRunner, gaRunner, mimicRunner]
    algorithm_name_list = ['RHC', 'SA', 'GA', 'MIMIC']
    fig, ax = plt.subplots(2,2, figsize=figsize)
    for algorithm, algorithm_name, best_params in zip(algorithm_list, algorithm_name_list, best_params_list):
        print(f'plot {algorithm_name}...')
        df_run_curves = fitness_per_iteration(algorithm, best_params)
        df_run_curves.index = range(len(df_run_curves))
        curve_list.append(df_run_curves)
        if len(df_run_curves) > longest:
            longest = len(df_run_curves)
            iter_s = df_run_curves['Iteration']
    for algorithm_name, df_run_curves in zip(algorithm_name_list, curve_list):   
        df_run_curves = df_run_curves.reindex(iter_s.index)
        df_run_curves['Iteration'] = iter_s
        df_run_curves = df_run_curves.ffill()
        ax[0,0].plot(df_run_curves['Iteration'], df_run_curves['Fitness'], label=algorithm_name)
        ax[0,1].plot(df_run_curves['Iteration'], df_run_curves['Time'], label=algorithm_name)
        ax[1,0].plot(df_run_curves['Iteration'], df_run_curves['FEvals'], label=algorithm_name)
    # ax[0,0].set_xscale('log')
    ax[0,0].set_xlabel('Iteration')
    ax[0,0].set_ylabel('Fitness')
    ax[0,0].title.set_text(f'Fitness on Iteration')
    ax[0,0].legend()
    # ax[0,1].set_xscale('log')
    ax[0,1].set_yscale('log')
    ax[0,1].set_xlabel('Iteration')
    ax[0,1].set_ylabel('Coverge Time (s)')
    ax[0,1].title.set_text(f'Time on Iteration')
    ax[0,1].legend()
    # ax[1,0].set_xscale('log')
    ax[1,0].set_yscale('log')
    ax[1,0].set_xlabel('Max Iteration')
    ax[1,0].set_ylabel('Function Evaluations')
    ax[1,0].title.set_text(f'Function Evalustions on Iteration')
    ax[1,0].legend()
    plt.tight_layout()
    plt.show()
        