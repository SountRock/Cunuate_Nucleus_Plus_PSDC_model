'''
Optimizer code and loss
'''

import numpy as np
import random
import csv
from deap import base, creator, tools, algorithms
from pymoo.indicators.hv import Hypervolume as HV
import json
import os
from scipy import signal
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv as _csv
import os
import json
import datetime
from concurrent.futures import ProcessPoolExecutor
import datetime, json, os, csv, random
from math import comb
from scipy.stats import pearsonr

'''
Base loss funcs (start learning)
'''
def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def rmse(y_pred, y_true):
    return np.sqrt(mse(y_pred, y_true))

def mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

'''
Huber loss: stronger penalty for outliers when delta < 1

Uses quadratic loss for residuals with absolute value <= delta and
linear loss beyond that. This makes the loss robust to outliers.
'''
def huber(y_pred, y_true, delta=0.35):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    r = y_true - y_pred
    abs_r = np.abs(r)
    mask = abs_r <= delta
    loss = np.where(mask, 0.5 * r**2, delta * (abs_r - 0.5 * delta))
    return np.mean(loss)


'''
Loss with motif similarity assessment

Combines RMSE with an additional multiplicative penalty based on a
motif dissimilarity score. If sequences are too short (<3), returns sqrt(MSE).
'''
def motiv_loss(y_pred, y_true, power=10): 
    if len(y_true) < 3 or len(y_pred) < 3:
        return  np.sqrt(mse(y_pred, y_true))
    motiv_penalty = dissimilarity_score(y_true, y_pred)
    rmse = np.sqrt(mse(y_pred, y_true))
    return rmse * (power * motiv_penalty) 


'''
Loss with motif similarity assessment v2

Returns RMSE scaled by an additive term proportional to motif penalty and
the maximum of the ground-truth signal. For short signals (<3), returns sqrt(MSE).
'''
def motiv_loss2(y_pred, y_true):
    if len(y_true) < 3 or len(y_pred) < 3:
        return  np.sqrt(mse(y_pred, y_true))
    motiv_penalty = dissimilarity_score(y_true, y_pred)
    rmse = np.sqrt(mse(y_pred, y_true))
    return rmse + motiv_penalty * np.max(y_true)

'''
Huber loss combined with motif penalty.

Computes the mean Huber loss and multiplies it by (1 + power * motif_penalty).
For short signals (<3) returns sqrt(MSE).
'''
def motiv_huber_loss(y_pred, y_true, power=10.0, delta=0.35):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if len(y_true) < 3 or len(y_pred) < 3:
        return  np.sqrt(mse(y_pred, y_true))

    r = y_true - y_pred
    abs_r = np.abs(r)
    mask = abs_r <= delta
    loss = np.where(mask, 0.5 * r**2, delta * (abs_r - 0.5*delta))
    huber_mean = np.mean(loss)

    motiv_penalty = dissimilarity_score(y_true, y_pred)

    return huber_mean * (1 + power * motiv_penalty)

'''
Combined RMSE + Huber + motif penalty (version 3).

Returns rmse + mean Huber loss + (power * motif_penalty * max(y_true)).
For short signals (<3) returns sqrt(MSE).
'''
def motiv_huber_loss_3(y_pred, y_true, power=5.0, delta=0.35):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if len(y_true) < 3 or len(y_pred) < 3:
        return  np.sqrt(mse(y_pred, y_true))

    r = y_true - y_pred
    abs_r = np.abs(r)
    mask = abs_r <= delta
    loss = np.where(mask, 0.5 * r**2, delta * (abs_r - 0.5*delta))

    huber_mean = np.mean(loss)
    rmse = np.sqrt(mse(y_pred, y_true))
    motiv_penalty = dissimilarity_score(y_true, y_pred)

    return rmse + huber_mean + (power * motiv_penalty * np.max(y_true))

'''
Huber loss with conditional motif scaling.

If y_true is short (<3), apply motif penalty multiplicatively to huber mean.
Otherwise return huber_mean only.
'''
def motiv_huber_loss_2(y_pred, y_true, power=10.0, delta=0.35):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    r = y_true - y_pred
    abs_r = np.abs(r)
    mask = abs_r <= delta
    loss = np.where(mask, 0.5 * r**2, delta * (abs_r - 0.5*delta))
    huber_mean = np.mean(loss)

    if len(y_true) < 3:
        motiv_penalty = dissimilarity_score(y_true, y_pred)
        return huber_mean * (1 + power * motiv_penalty)
    else: 
        return huber_mean

'''
Correlation-aware composite loss
'''
def corr_loss(y_pred, y_true,
                     delta_huber=0.35,
                     quantil_huber=0.75,
                     global_weight=2.0,
                     local_weight=1.0,
                     zero_k=10.0,
                     zero_max=1.0,
                     zero_mid=0.5):
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    if y_pred.size < 3 or y_true.size < 3:
        diff = y_true - y_pred
        return float(np.sqrt(np.mean(diff ** 2)))

    # Huber (fixed delta)
    r = y_true - y_pred
    abs_r = np.abs(r)
    mask = abs_r <= delta_huber
    huber = np.where(mask, 0.5 * r**2, delta_huber * (abs_r - 0.5 * delta_huber))

    huber_median = float(np.quantile(huber, quantil_huber))
    max_huber = float(np.max(huber))       

    # Global correlation (pearsonr)
    rho_global = float(pearsonr(y_true, y_pred)[0])
    if not np.isfinite(rho_global):
        rho_global = 0.0
    global_penalty = 1.0 - abs(rho_global)  # in [0,1]

    # Local / lag correlation via cross-correlation
    yt_c = y_true - y_true.mean()
    yp_c = y_pred - y_pred.mean()
    cross = signal.correlate(yt_c, yp_c, mode='full')
    std_t = y_true.std(ddof=0)
    std_p = y_pred.std(ddof=0)
    denom = float(max(1e-12, y_true.size * std_t * std_p))
    cross_norm = cross / denom
    max_abs_corr = float(np.max(np.abs(cross_norm)))
    max_abs_corr = min(1.0, max_abs_corr)
    local_penalty = 1.0 - max_abs_corr

    # Combined anti-correlation factor with weights
    corr_factor = float(global_weight * global_penalty + local_weight * local_penalty)
    max_abs_diff = float(np.max(np.abs(y_true - y_pred)))
    anticorr_component = corr_factor * max_abs_diff

    # Sigmoidal penalty for zeros in prediction
    N = y_pred.size
    count_zeros = int(np.sum(np.isclose(y_pred, 0.0)))
    x_pos = float(count_zeros)
    x_half = float(zero_mid) * float(N)
    smooth = -float(zero_k)
    s = 1.0 / (1.0 + np.exp(smooth * (x_pos - x_half)))
    zero_penalty = float(zero_max * s)

    final = huber_median * (1.0 + max_huber * anticorr_component + zero_penalty)
    return float(final)
    

'''
Motif similarity assessment

Compute motif similarity score via normalized cross-correlation.
Returns maximum absolute normalized cross-correlation between y_pred and y_true.
If either vector has zero norm, returns 1.0 (max dissimilarity / fallback).
'''
def dissimilarity_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    a = y_true - np.mean(y_true)
    b = y_pred - np.mean(y_pred)

    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0

    cc = signal.correlate(b, a, mode='full')
    norm_cc = np.abs(cc) / (na * nb + 1e-16)
    max_corr = float(np.nanmax(norm_cc))
    return max_corr 


'''
Normalization if spectra are in different ranges
'''
def normalize(arr):
    return np.log1p(arr)

#==============================================================================
'''
Multi-objective optimization using NSGA (Pareto optimization) + initialization of initial parameters
Because Optuna is too weak, pymoo does almost the same thing, but takes much longer. DEAP is the best option.
P.S. Optuna is really trash

Arguments:
    target_funcs: list of target functions
    t_matrix: list of time arrays (x-axes of the target functions) for each target function
    y_true_matrix: list of reference data (ground truth) for each target function
    param_bounds: dictionary {param_name: (low, high)} specifying parameter bounds.
    loss_func: loss function
    n_gen: number of generations (if None -> determined automatically)
    pop_size: population size
    algorithm_ver: 2 - NSGAII, 3 - NSGAIII, default: 2 (RVEA and other algorithms were not effective enough)
    log_path: path for CSV logging
    checkpoint_path: path for saving/restoring the state
    checkpoint_every: how often to save a checkpoint (in generations)
    priorities: weights for each target function (loss penalty). Simple and more effective than more complex alternatives
    thread_log_path: path for logging parallel computations (for monitoring/debugging)
    n_workers: number of parallel worker threads
    anti_dup_strategy: strategy for handling duplicates:
        - none: do nothing with duplicates
        - delete: remove duplicates within a single generation
                  (duplicates are detected using an identical key formed from loss values)
        - remutate: force duplicates to mutate again until they obtain a unique loss-key,
                    or until the maximum number of attempts is exhausted (max_mut_attempts)

    A checkpoint is a JSON file. Why JSON and not pickle?
    Because JSON makes it easier to quickly inspect whether the optimization is proceeding correctly
    and whether NaN or Inf values are appearing.
    Also, the checkpoint size is not large enough to justify compression.
'''
def run_nsga_optimization_threads(mega_func, t_matrix, y_true_matrix, param_bounds: dict,
                                   loss_func=mse, n_gen=None, pop_size=120, algorithm_ver=2,
                                   log_path=None, checkpoint_path=None, checkpoint_every=5, max_mut_attempts=10,
                                   priorities=None, init_params=None, delta_init_params=None,
                                   n_workers=None, thread_log_path="threads_log.txt", anti_dup_strategy='none'):

    # number of objective functions (one per target function in y_true_matrix)
    num_objectives = len(y_true_matrix)
    if len(t_matrix) != num_objectives:
        raise ValueError("t_matrix and y_true_matrix must contain the same number of functions")

    all_param_names = list(param_bounds.keys())
    bounds = [param_bounds[name] for name in all_param_names]

    if n_gen is None:
        n_gen = max(80, 15 * len(bounds))

    if priorities is None:
        priorities = [1.0] * num_objectives
    elif len(priorities) != num_objectives:
        raise ValueError("Length of priorities must match the number of objective functions")

    # Create DEAP fitness and individual classes if not present
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,) * num_objectives)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    # register attribute generators for each parameter within bounds
    for i, (lo, hi) in enumerate(bounds):
        toolbox.register(f"attr_{i}", random.uniform, lo, hi)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     [toolbox.__getattribute__(f"attr_{i}") for i in range(len(bounds))], n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluate with support for parallelism (wraps parallel_evaluate)
    def evaluate(ind):
        ind_values = [ind[all_param_names.index(p)] for p in all_param_names]
        losses, deviations = parallel_evaluate(ind_values, t_matrix, y_true_matrix,
                                               mega_func, loss_func, priorities, thread_log_path)
        ind.deviations = deviations
        return tuple(losses)
    
    def _sanitize_dev_pair(pair):
        if pair is None:
            return [None, None]
        a, b = pair
        a_s = None if (a is None or not np.isfinite(a)) else float(a)
        b_s = None if (b is None or not np.isfinite(b)) else float(b)
        return [a_s, b_s]

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=[lo for lo, hi in bounds],
                     up=[hi for lo, hi in bounds],
                     eta=15.0)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=[lo for lo, hi in bounds],
                     up=[hi for lo, hi in bounds],
                     eta=20.0,
                     indpb=0.2)
    
    match algorithm_ver:
        case 2:
            toolbox.register("select", tools.selNSGA2)
        case 3:
            # Reference points for NSGA-III
            ref_points = generate_ref_points(num_objectives, pop_size)
            toolbox.register("select", tools.selNSGA3WithMemory(ref_points))
        case _:
            toolbox.register("select", tools.selNSGA2)
 
    reference_point = [1.2] * num_objectives
    hv_indicator = HV(ref_point=reference_point)

    # Checkpoint loading or initialization
    start_gen = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        start_gen = checkpoint["generation"] + 1

        # restore population and fitness values
        population = [creator.Individual(ind) for ind in checkpoint["population"]]
        for ind, fit in zip(population, checkpoint["fitnesses"]):
            ind.fitness.values = tuple(fit)

        # restore deviations (if present) — safely and aligned to the number of objectives
        devs_all = checkpoint.get("deviations", None)
        if devs_all is None:
            default_dev = [(float('nan'), float('nan'))] * num_objectives
            for ind in population:
                ind.deviations = list(default_dev)
        else:
            for ind, dev_list in zip(population, devs_all):
                devs_reconstructed = []
                if dev_list is None:
                    dev_list = []
                for i in range(num_objectives):
                    if i < len(dev_list):
                        pair = dev_list[i]
                        if pair is None:
                            a, b = float('nan'), float('nan')
                        else:
                            a = pair[0]
                            b = pair[1]
                            a = float(a) if (a is not None) else float('nan')
                            b = float(b) if (b is not None) else float('nan')
                        devs_reconstructed.append((a, b))
                    else:
                        devs_reconstructed.append((float('nan'), float('nan')))
                ind.deviations = devs_reconstructed

        print(f"Checkpoint restored at start_gen={start_gen}")
    else:
        if init_params is not None and delta_init_params is not None:
            population = []
            exact = creator.Individual([init_params[name] for name in all_param_names])
            population.append(exact)

            while len(population) < pop_size:
                indiv_vals = [
                    init_params[name] * (1 + random.uniform(-delta_init_params, delta_init_params))
                    for name in all_param_names
                ]
                population.append(creator.Individual(indiv_vals))
        else:
            population = toolbox.population(n=pop_size)

        for i in range(len(population)):
            ind = population[i]
            fitness = toolbox.evaluate(ind)
            # guarantee the presence of deviations after evaluate
            if not hasattr(ind, 'deviations'):
                ind.deviations = [(float('nan'), float('nan'))] * num_objectives

            if any(np.isnan(v) or np.isinf(v) for v in fitness):
                if init_params is not None:
                    replacement = creator.Individual([init_params[name] for name in all_param_names])
                    replacement_fitness = toolbox.evaluate(replacement)
                    replacement.fitness.values = tuple(replacement_fitness)
                    if not hasattr(replacement, 'deviations'):
                        replacement.deviations = [(float('nan'), float('nan'))] * num_objectives
                    population[i] = replacement
                else:
                    # as a last resort, mark NaNs and create default deviations
                    ind.fitness.values = tuple(float('nan') for _ in range(num_objectives))
                    ind.deviations = [(float('nan'), float('nan'))] * num_objectives
                    population[i] = ind
            else:
                ind.fitness.values = fitness


    # CSV logging setup
    if log_path:
        mode = 'w' if start_gen == 0 else 'a'
        log_file = open(log_path, mode=mode, newline='')
        csv_writer = csv.writer(log_file)
        if start_gen == 0:
            header = ['generation']
            for i in range(num_objectives):
                header += [f'tf{i+1}_loss', f'tf{i+1}_min_dev', f'tf{i+1}_max_dev']
            header += all_param_names + ['hypervolume']
            csv_writer.writerow(header)
    else:
        csv_writer = None

    prev_hv = 0.0

    # Main evolutionary loop
    for gen in range(start_gen, n_gen):
        # compute current Pareto front and hypervolume
        front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        fits_np = np.array([ind.fitness.values for ind in front])
        current_hv = hv_indicator.do(fits_np)
        hv_change = current_hv - prev_hv
        prev_hv = current_hv

        # Adaptive crossover / mutation probabilities based on HV change
        # Hypervolume (HV) reflects the area (or volume) covered by the Pareto front
        # when considering all objectives simultaneously.
        # If HV increases -> the Pareto front improves:
        #   1) Increase crossover probability (cxpb) to combine good solutions more actively
        #   2) Decrease mutation probability (mutpb) to reduce random perturbations
        # If HV decreases or does not change -> the population stagnates:
        #   Increase mutation and reduce crossover to encourage exploration of new regions
        cxpb = min(0.9, 0.6 + 0.3 * hv_change) if hv_change > 0 else 0.4
        mutpb = max(0.1, 0.4 - 0.3 * hv_change) if hv_change > 0 else 0.6
        # The parameters are selected according to practice

        # Variation
        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)

        # Evaluate invalid offspring (parallel if requested)
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        if n_workers and n_workers > 1:
            evaluate_population_parallel(invalid_inds, n_workers, t_matrix, y_true_matrix, mega_func,
                                         loss_func, priorities, all_param_names, thread_log_path)
        else:
            for ind in invalid_inds:
                ind.fitness.values = toolbox.evaluate(ind)

        # Selection for the next generation
        population = toolbox.select(population + offspring, k=pop_size)

        # *** Anti-duplication block (strategies: 'remutate'|'delete'|'none') ***
        # Determine strategy 
        if anti_dup_strategy == 'none':
            # Do nothing: keep population as-is
            pass
        else:
            # Form numeric keys from losses (rounded to reduce FP noise)
            existing_keys = set()
            key_to_inds = {}
            for idx, ind in enumerate(population):
                key = tuple(np.round(np.asarray(ind.fitness.values, dtype=float), 8))
                key_to_inds.setdefault(key, []).append(idx)
                existing_keys.add(key)

            # For each group of duplicates (len>1) try to make all but one unique
            for key, indices in list(key_to_inds.items()):
                if len(indices) <= 1:
                    continue

                # keep the first, process the rest
                for dup_idx in indices[1:]:
                    if anti_dup_strategy == 'delete':
                        # simple replacement: create a fresh individual and evaluate it
                        new_ind = toolbox.individual()
                        try:
                            new_ind.fitness.values = toolbox.evaluate(new_ind)
                        except Exception:
                            new_ind.fitness.values = tuple([1e6] * len(y_true_matrix))
                        new_key = tuple(np.round(np.asarray(new_ind.fitness.values, dtype=float), 8))
                        population[dup_idx] = new_ind
                        existing_keys.add(new_key)
                        key_to_inds.setdefault(new_key, []).append(dup_idx)
                        continue  # next duplicate

                    # anti_dup_strategy == 'remutate' : try mutating the duplicate in-place
                    ind = population[dup_idx]
                    attempts = 0
                    unique_obtained = False
                    while attempts < max_mut_attempts:
                        attempts += 1
                        # mutate in-place 
                        ind, = toolbox.mutate(ind)
                        # invalidate previous fitness and re-evaluate
                        try:
                            del ind.fitness.values
                        except Exception:
                            pass
                        try:
                            ind.fitness.values = toolbox.evaluate(ind)
                        except Exception:
                            ind.fitness.values = tuple([1e6] * len(y_true_matrix))

                        new_key = tuple(np.round(np.asarray(ind.fitness.values, dtype=float), 8))
                        if new_key not in existing_keys:
                            # success — register and replace
                            existing_keys.add(new_key)
                            key_to_inds.setdefault(new_key, []).append(dup_idx)
                            population[dup_idx] = ind
                            unique_obtained = True
                            break

                    if not unique_obtained:
                        # exhausted attempts: keep last mutated individual (may still be duplicate)
                        population[dup_idx] = ind

        # Logging to CSV if enabled
        if csv_writer:
            best_inds = tools.sortNondominated(population, k=pop_size, first_front_only=True)[0]
            for best_ind in best_inds:
                row = [gen]
                for loss, (min_dev, max_dev) in zip(best_ind.fitness.values, best_ind.deviations):
                    row += [loss, min_dev, max_dev]
                row += list(best_ind) + [current_hv]
                csv_writer.writerow(row)

        # Periodically save checkpoint
        if checkpoint_path and checkpoint_every and (gen % checkpoint_every == 0):
            checkpoint_data = {
                "generation": gen,
                "population": [list(ind) for ind in population],
                "fitnesses": [list(ind.fitness.values) for ind in population],
                "deviations": [
                    [ _sanitize_dev_pair(pair) for pair in getattr(ind, 'deviations', [(None, None)] * num_objectives) ]
                    for ind in population
                ],
                "param_names": all_param_names,
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f)

    if csv_writer:
        log_file.close()

    # Extract Pareto front and prepare results (clamp parameters to bounds)
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    results = []
    for ind in pareto_front:
        clamped = [max(lo, min(ind[i], hi)) for i, (lo, hi) in enumerate(bounds)]
        results.append((clamped, ind.fitness.values, getattr(ind, 'deviations', None)))

    return all_param_names, results, pareto_front

'''
Function for parallel evaluate
'''
def parallel_evaluate(ind_values, t_matrix, y_true_matrix, mega_func, loss_func, priorities, thread_log_path=None):
    start_time = datetime.datetime.now()
    num_inds = 1  # always one individual per thread
    if thread_log_path:
        with open(thread_log_path, 'a') as f:
            f.write(f"[{start_time}] Thread started for {num_inds} individual(s): {ind_values}\n")

    try:
        y_pred_matrix = mega_func(t_matrix, *ind_values)
    except Exception as e:
        if thread_log_path:
            with open(thread_log_path, 'a') as f:
                f.write(f"[{datetime.datetime.now()}] Error evaluating {ind_values}: {e}\n")
        losses, deviations = tuple([1e6]*len(y_true_matrix)), [(0.0, 0.0)]*len(y_true_matrix)
    else:
        losses, deviations = [], []
        for i, y_true in enumerate(y_true_matrix):
            y_pred_s = np.asarray(y_pred_matrix[i])
            y_true_s = np.asarray(y_true)
            y_pred = normalize(y_pred_s)
            y_true_n = normalize(y_true_s)
            loss_val = loss_func(y_pred, y_true_n)
            weighted_loss = loss_val * priorities[i]
            losses.append(weighted_loss)
            diff = y_pred_s - y_true_s
            deviations.append((float(np.min(diff)), float(np.max(diff))))

    end_time = datetime.datetime.now()
    if thread_log_path:
        with open(thread_log_path, 'a') as f:
            f.write(f"[{end_time}] Thread finished for {num_inds} individual(s): {ind_values}, Losses: {losses}\n")
    return tuple(losses), deviations

'''
Function for parallel evaluation of a population
'''
def evaluate_population_parallel(population, n_workers, t_matrix, y_true_matrix, mega_func,
                                 loss_func, priorities, all_param_names, thread_log_path):
    futures_map = {}  
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # submit all tasks to the pool
        for ind in population:
            ind_values = [ind[all_param_names.index(p)] for p in all_param_names]
            fut = executor.submit(parallel_evaluate, ind_values, t_matrix, y_true_matrix,
                                  mega_func, loss_func, priorities, thread_log_path)
            futures_map[fut] = ind

        # collect results as they complete
        for fut in as_completed(futures_map):
            ind = futures_map[fut]
            try:
                losses, deviations = fut.result()
            except Exception as e:
                with open(thread_log_path, 'a') as f:
                    f.write(f"[{datetime.datetime.now()}] Unexpected exception for individual {ind}: {e}\n")
                losses = tuple([1e6] * len(y_true_matrix))
                deviations = [(0.0, 0.0)] * len(y_true_matrix)
            ind.fitness.values = losses
            ind.deviations = deviations

'''
For NSGAIII. Generation of search space points
'''
def generate_ref_points(nobj, pop_size):
    p = 1
    while comb(p + nobj - 1, nobj - 1) < pop_size:
        p += 1
    return tools.uniform_reference_points(nobj, p)







