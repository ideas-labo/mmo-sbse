import os
import csv
import sys
import random
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau, entropy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances
import networkx as nx

try:
    from jmetal.core.quality_indicator import HyperVolume
    from jmetal.util.ranking import FastNonDominatedRanking
except Exception:
    HyperVolume = None
    FastNonDominatedRanking = None

sys.path.append('../')
sys.path.append('../..')
sys.path.append('../../..')


def is_pareto_optimal_minimize(point, points):
    point = tuple(point) if isinstance(point, (list, np.ndarray)) else point
    points = [tuple(p) if isinstance(p, (list, np.ndarray)) else p for p in points]

    is_pareto = True
    for other_point in points:
        other_point = tuple(other_point) if isinstance(other_point, (list, np.ndarray)) else other_point
        condition1 = other_point[0] <= point[0] and other_point[1] < point[1]
        condition2 = other_point[0] < point[0] and other_point[1] <= point[1]
        if condition1 or condition2:
            is_pareto = False
            break
    return is_pareto


def infer_unified_dimensions(sampled_data):
    if not sampled_data:
        return 1, 1
    max_prod = 0
    max_feat = 0
    for ind in sampled_data:
        if isinstance(ind, (list, tuple, np.ndarray)):
            prod_count = len(ind)
            if prod_count > max_prod:
                max_prod = prod_count
            for p in ind:
                if isinstance(p, (list, tuple, np.ndarray)):
                    feat_count = len(p)
                    if feat_count > max_feat:
                        max_feat = feat_count
    return max_prod if max_prod > 0 else 1, max_feat if max_feat > 0 else 1


def unify_individual_dimension(individual, product_count, feature_count):
    if isinstance(individual, np.ndarray):
        individual = individual.tolist()
    if not isinstance(individual, (list, tuple)):
        individual = [individual]
    individual = list(individual)
    if len(individual) < product_count:
        for _ in range(product_count - len(individual)):
            individual.append([0] * feature_count)
    elif len(individual) > product_count:
        individual = individual[:product_count]
    unified = []
    for prod in individual:
        if isinstance(prod, np.ndarray):
            prod = prod.tolist()
        if not isinstance(prod, (list, tuple)):
            prod = [prod]
        prod = list(prod)
        if len(prod) < feature_count:
            prod = prod + [0] * (feature_count - len(prod))
        elif len(prod) > feature_count:
            prod = prod[:feature_count]
        unified.append(tuple(prod))
    return tuple(unified)


def calculate_java_hamming_distance(products1, products2):
    if products1 is None or products2 is None:
        return int(np.iinfo(np.int32).max)
    distance = 0
    all_min_lens = []
    for p1 in products1:
        for p2 in products2:
            min_len = min(len(p1), len(p2))
            all_min_lens.append(min_len)
            for i in range(min_len):
                if p1[i] != p2[i]:
                    distance += 1
    size_diff = abs(len(products1) - len(products2))
    avg_feat_len = int(np.mean(all_min_lens)) if all_min_lens else 0
    distance += size_diff * avg_feat_len
    return distance


class ExactHammingIndex:
    def __init__(self):
        self.product_count = None
        self.feature_count = None
        self.data = []
        self._position_index = defaultdict(list)

    def add(self, vectors):
        vecs = list(vectors)
        if self.product_count is None or self.feature_count is None:
            pc, fc = infer_unified_dimensions(vecs)
            self.product_count = pc
            self.feature_count = fc
        start = len(self.data)
        for vec in vecs:
            unified = unify_individual_dimension(vec, self.product_count, self.feature_count)
            self.data.append(unified)
        for idx in range(start, len(self.data)):
            indiv = self.data[idx]
            for p in range(self.product_count):
                prod = indiv[p]
                for f in range(self.feature_count):
                    v = prod[f]
                    self._position_index[(p, f, v)].append(idx)

    def search(self, query, k, max_candidates=1000):
        if not self.data:
            return [], []
        q_unified = unify_individual_dimension(query, self.product_count, self.feature_count)
        candidate_counts = defaultdict(int)
        for p in range(self.product_count):
            prod = q_unified[p]
            for f in range(self.feature_count):
                v = prod[f]
                for idx in self._position_index.get((p, f, v), []):
                    candidate_counts[idx] += 1
        sorted_cands = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)
        cand_indices = [idx for idx, _ in sorted_cands[:max_candidates]]
        if not cand_indices:
            return [], []
        distances = []
        for idx in cand_indices:
            target = self.data[idx]
            distances.append(calculate_java_hamming_distance(q_unified, target))
        sorted_idx = np.argsort(distances)[:k]
        nearest_idxs = [cand_indices[i] for i in sorted_idx]
        nearest_vecs = [self.data[i] for i in nearest_idxs]
        nearest_dists = [distances[i] for i in sorted_idx]
        return nearest_vecs, nearest_dists


def get_adaptive_k(dimension):
    if dimension <= 50:
        return 20
    elif dimension <= 200:
        return 10
    else:
        return 5


class NeighborFinder:
    def __init__(self):
        self.index = None

    def generate_neighbor_solutions(self, solution, sampled_data, random_seed=None):
        if isinstance(sampled_data, np.ndarray):
            sampled_data = sampled_data.tolist()
        if self.index is None:
            self.index = ExactHammingIndex()
            self.index.add(sampled_data)
        total_dim = self.index.product_count * self.index.feature_count
        k = get_adaptive_k(total_dim)
        neighbors, _ = self.index.search(solution, k=k)
        return neighbors


neighbor_finder = NeighborFinder()


def generate_neighbor_solutions(solution, sampled_data, random_seed=None):
    return neighbor_finder.generate_neighbor_solutions(solution, sampled_data, random_seed)


def find_local_efficient_points(decision_vars, objective_values, sampled_data, random_seed=None):
    if isinstance(decision_vars, np.ndarray):
        decision_vars = decision_vars.tolist()
    if isinstance(sampled_data, np.ndarray):
        sampled_data = sampled_data.tolist()
    if len(decision_vars) < 2:
        return np.array([])
    var_to_obj = {tuple(v): tuple(o) for v, o in zip(decision_vars, objective_values)}
    sampled_set = sampled_data
    local_flags = []
    for i, var in enumerate(decision_vars):
        cur_obj = tuple(objective_values[i])
        dominated = False
        neighbors = generate_neighbor_solutions(var, sampled_set, random_seed)
        for n in neighbors:
            if n in var_to_obj:
                nobj = var_to_obj[n]
                if (nobj[0] <= cur_obj[0] and nobj[1] < cur_obj[1]) or (nobj[0] < cur_obj[0] and nobj[1] <= cur_obj[1]):
                    dominated = True
                    break
        local_flags.append(not dominated)
    return np.where(local_flags)[0]


def find_local_efficient_points_literature(decision_vars, objective_values, sampled_data, random_seed=None):
    return find_local_efficient_points(decision_vars, objective_values, sampled_data, random_seed=random_seed)


def calculate_connected_components(decision_vars, local_efficient_indices, sampled_data, random_seed=None):
    if len(local_efficient_indices) < 1:
        return 0
    if isinstance(decision_vars, np.ndarray):
        decision_vars = decision_vars.tolist()
    sampled_set = sampled_data if not isinstance(sampled_data, np.ndarray) else sampled_data.tolist()
    le_vars = [decision_vars[i] for i in local_efficient_indices]
    idx_map = {tuple(v): i for i, v in enumerate(le_vars)}
    G = nx.Graph()
    for v in le_vars:
        G.add_node(idx_map[tuple(v)])
        neighbors = generate_neighbor_solutions(v, sampled_set, random_seed)
        for n in neighbors:
            if n in idx_map:
                G.add_edge(idx_map[tuple(v)], idx_map[n])
    return len(list(nx.connected_components(G)))


def find_global_pareto_points_simple(objective_values):
    if len(objective_values) < 2:
        return np.array([], dtype=int)
    objs = np.array(objective_values)
    pareto_idx = []
    for i in range(len(objs)):
        cur = objs[i]
        dominated = False
        for j in range(len(objs)):
            if i == j:
                continue
            other = objs[j]
            if (other[0] <= cur[0] and other[1] < cur[1]) or (other[0] < cur[0] and other[1] <= cur[1]):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(i)
    return np.array(pareto_idx, dtype=int)


def calculate_pareto_connected_components(decision_vars, objective_values, sampled_data, random_seed=None):
    if isinstance(decision_vars, np.ndarray):
        decision_vars = decision_vars.tolist()
    if isinstance(sampled_data, np.ndarray):
        sampled_data = sampled_data.tolist()
    pareto_idx = find_global_pareto_points_simple(objective_values)
    if len(pareto_idx) < 1:
        return 0
    pareto_vars = [decision_vars[i] for i in pareto_idx]
    idx_map = {tuple(v): i for i, v in enumerate(pareto_vars)}
    G = nx.Graph()
    for v in pareto_vars:
        G.add_node(idx_map[tuple(v)])
        neighbors = generate_neighbor_solutions(v, sampled_data, random_seed)
        for n in neighbors:
            if n in idx_map:
                G.add_edge(idx_map[tuple(v)], idx_map[n])
    return len(list(nx.connected_components(G)))


def calculate_dominance_change_frequency(sampled_dict, random_seed=None):
    if len(sampled_dict) < 2:
        return 0.0
    decision_vars = list(sampled_dict.keys())
    objective_values = np.array(list(sampled_dict.values()))
    if decision_vars:
        neighbor_finder.generate_neighbor_solutions(decision_vars[0], decision_vars, random_seed)
    is_p = np.array([is_pareto_optimal_minimize(p, objective_values) for p in objective_values])
    var_to_idx = {tuple(v): i for i, v in enumerate(decision_vars)}
    changes = 0
    pairs = 0
    for i, v in enumerate(decision_vars):
        neighbors = generate_neighbor_solutions(v, decision_vars, random_seed)
        for n in neighbors:
            j = var_to_idx.get(n)
            if j is None or j == i:
                continue
            if is_p[i] != is_p[j]:
                changes += 1
            pairs += 1
    return changes / pairs if pairs > 0 else 0.0


def calculate_dominance_features(decision_vars, objective_values):
    if len(objective_values) < 2:
        return {'median_dominance': 0.0, 'max_dominance': 0}
    objs = np.array(objective_values)
    counts = []
    for i in range(len(objs)):
        cnt = 0
        cur = objs[i]
        for j in range(len(objs)):
            if i == j:
                continue
            other = objs[j]
            if (other[0] <= cur[0] and other[1] < cur[1]) or (other[0] < cur[0] and other[1] <= cur[1]):
                cnt += 1
        counts.append(cnt)
    return {'median_dominance': float(np.median(counts)), 'max_dominance': int(np.max(counts))}


def calculate_local_to_global_ratio(decision_vars, objective_values, sampled_data, random_seed=None):
    if isinstance(decision_vars, np.ndarray):
        decision_vars = decision_vars.tolist()
    if isinstance(sampled_data, np.ndarray):
        sampled_data = sampled_data.tolist()
    global_idx = find_global_pareto_points_simple(objective_values)
    local_idx = find_local_efficient_points(decision_vars, objective_values, sampled_data, random_seed)
    gset = set(global_idx)
    lset = set(local_idx.tolist())
    inter = gset.intersection(lset)
    return {'local_as_global_ratio': float(len(inter) / len(lset)) if len(lset) > 0 else 0.0,
            'num_global': len(gset), 'num_local': len(lset), 'num_intersection': len(inter),
            'global_not_local': len(gset - lset)}


def calculate_avg_discrete_descent_cone(decision_vars, objective_values, sampled_data, random_seed=None):
    if isinstance(decision_vars, np.ndarray):
        decision_vars = decision_vars.tolist()
    if isinstance(sampled_data, np.ndarray):
        sampled_data = sampled_data.tolist()
    if len(decision_vars) == 0:
        return 0.0
    var_to_obj = {tuple(v): tuple(o) for v, o in zip(decision_vars, objective_values)}
    sampled_set = sampled_data
    total = 0.0
    valid = 0
    for var in decision_vars:
        neighbors = generate_neighbor_solutions(var, sampled_set, random_seed)
        if not neighbors:
            continue
        cur_obj = var_to_obj[tuple(var)]
        dom = 0
        for n in neighbors:
            if n in var_to_obj:
                nobj = var_to_obj[n]
                if (nobj[0] <= cur_obj[0] and nobj[1] < cur_obj[1]) or (nobj[0] < cur_obj[0] and nobj[1] <= cur_obj[1]):
                    dom += 1
        total += dom / len(neighbors)
        valid += 1
    return float(total / valid) if valid > 0 else 0.0


def create_grid(objective_values, step=0.2, feature_range=(-2, 2)):
    if len(objective_values) == 0:
        return None, None
    min_val, max_val = feature_range
    grid_points = np.arange(min_val, max_val + step, step)
    grid_indices = []
    for obj in objective_values:
        i = np.clip(np.searchsorted(grid_points, obj[0], side='right') - 1, 0, len(grid_points) - 2)
        j = np.clip(np.searchsorted(grid_points, obj[1], side='right') - 1, 0, len(grid_points) - 2)
        grid_indices.append((int(i), int(j)))
    return {'size': len(grid_points)}, np.array(grid_indices)


def calculate_grid_features(decision_vars, objective_values, is_pareto_list):
    if isinstance(decision_vars, np.ndarray):
        decision_vars = decision_vars.tolist()
    grid_params, grid_indices = create_grid(objective_values)
    if grid_params is None:
        return {'grid_density_var': 0.0, 'pareto_grid_ratio': 0.0, 'adjacent_grid_hamming_mean': 0.0}
    grid_size = grid_params['size']
    grid_counts = {}
    for idx in grid_indices:
        grid_counts[tuple(idx)] = grid_counts.get(tuple(idx), 0) + 1
    count_vals = np.array(list(grid_counts.values())) if grid_counts else np.array([0.0])
    density_var = float(np.var(count_vals)) if count_vals.size > 0 else 0.0
    unique = set(tuple(idx) for idx in grid_indices)
    pareto_grids = set(tuple(grid_indices[i]) for i in range(len(is_pareto_list)) if is_pareto_list[i])
    pareto_ratio = float(len(pareto_grids) / len(unique)) if len(unique) > 0 else 0.0
    adj_pairs = 0
    total_hamming = 0
    for i in range(len(decision_vars)):
        cur_grid = tuple(grid_indices[i])
        cur_dec = decision_vars[i]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj = (cur_grid[0] + dx, cur_grid[1] + dy)
            if 0 <= adj[0] < (grid_size - 1) and 0 <= adj[1] < (grid_size - 1):
                adj_idxs = [j for j in range(len(decision_vars)) if tuple(grid_indices[j]) == adj]
                if adj_idxs:
                    adj_dec = decision_vars[adj_idxs[0]]
                    total_hamming += calculate_java_hamming_distance(cur_dec, adj_dec)
                    adj_pairs += 1
    adj_mean = float(total_hamming / adj_pairs) if adj_pairs > 0 else 0.0
    return {'grid_density_var': density_var, 'pareto_grid_ratio': pareto_ratio, 'adjacent_grid_hamming_mean': adj_mean}


def calculate_hypervolume(solutions, ref_point):
    if HyperVolume is None:
        return 0.0
    hv = HyperVolume(reference_point=ref_point)
    return hv.compute(solutions)


def calculate_level_count(points):
    if FastNonDominatedRanking is None:
        return 0
    ranking = FastNonDominatedRanking()
    solutions = [type('S', (object,), {'objectives': p, 'attributes': {}}) for p in points]
    ranking.compute_ranking(solutions)
    return ranking.get_number_of_subfronts()


def calculate_objective_variance(points, dict_search):
    if points is None or len(points) == 0:
        return 0.0
    vars_ = np.var(points, axis=0)
    return float(np.mean(vars_))


def calculate_objective_entropy(points, dict_search):
    if points is None or len(points) == 0:
        return 0.0
    ent = []
    for i in range(points.shape[1]):
        _, counts = np.unique(points[:, i], return_counts=True)
        ent.append(float(entropy(counts)))
    return float(np.mean(ent))


def print_and_save_statistics(sampled_data, sampled_dict, r_pareto_count, total_points_count,
                              fig_num, x, y, random_seed, sampling_method, sample_size,
                              dataset_name, mode, unique_id, sample_type,
                              dominance_change_frequency, reverse, dag=None, num_valid_tasks=None):
    if isinstance(sampled_data, np.ndarray):
        sampled_data = sampled_data.tolist()

    adjacent_grid_hamming_mean = 0.0
    average_hamming_distance = 0.0
    avg_discrete_descent_cone = 0.0
    connected_components_pareto = 0
    connected_components = 0
    dominance_change_frequency_stat = float(dominance_change_frequency) if isinstance(dominance_change_frequency, (int, float)) else dominance_change_frequency
    grid_density_var = 0.0
    hypervolume = 0.0
    kendalltau_corr = 0.0
    level_count = 0
    local_efficient_points_count = 0
    local_to_global_ratio = 0.0
    max_dominance_count = 0
    median_dominance_count = 0.0
    objective_function_value_entropy = 0.0
    objective_function_value_variance = 0.0
    pareto_grid_ratio = 0.0
    pearson_corr = 0.0
    slope = 0.0
    spearman_corr = 0.0
    pareto_ratio = total_points_count.get('pareto', 0) / (total_points_count.get('pareto', 0) + total_points_count.get('non_pareto', 0)) if (total_points_count.get('pareto', 0) + total_points_count.get('non_pareto', 0)) > 0 else 0.0

    if len(x) >= 2 and len(y) >= 2:
        decision_vars = list(sampled_dict.keys())
        objective_values = np.array(list(sampled_dict.values()))
        ref_point = np.array([np.max(x), np.max(y)])

        spearman_corr, _ = spearmanr(x, y)
        pearson_corr, _ = pearsonr(x, y)
        kendalltau_corr, _ = kendalltau(x, y)
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        slope = float(model.coef_[0])

        level_count = calculate_level_count(np.vstack([x, y]).T)
        objective_function_value_variance = calculate_objective_variance(np.vstack([x, y]).T, sampled_dict)
        objective_function_value_entropy = calculate_objective_entropy(np.vstack([x, y]).T, sampled_dict)

        avg_discrete_descent_cone = calculate_avg_discrete_descent_cone(decision_vars, objective_values, sampled_data, random_seed=random_seed)

        local_idx_lit = find_local_efficient_points_literature(decision_vars, objective_values, sampled_data, random_seed=random_seed)
        local_efficient_points_count = int(len(local_idx_lit))

        local_idx_dom = find_local_efficient_points(decision_vars, objective_values, sampled_data, random_seed=random_seed)
        connected_components = calculate_connected_components(decision_vars, local_idx_dom, sampled_data, random_seed=random_seed)
        connected_components_pareto = calculate_pareto_connected_components(decision_vars, objective_values, sampled_data, random_seed=random_seed)

        dom_feats = calculate_dominance_features(decision_vars, objective_values)
        max_dominance_count = int(dom_feats['max_dominance'])
        median_dominance_count = float(dom_feats['median_dominance'])

        l2g = calculate_local_to_global_ratio(decision_vars, objective_values, sampled_data, random_seed=random_seed)
        local_to_global_ratio = float(l2g['local_as_global_ratio'])

        is_pareto_list = np.array([is_pareto_optimal_minimize(p, objective_values) for p in objective_values])
        grid_feats = calculate_grid_features(decision_vars, objective_values, is_pareto_list)
        grid_density_var = float(grid_feats['grid_density_var'])
        pareto_grid_ratio = float(grid_feats['pareto_grid_ratio'])
        adjacent_grid_hamming_mean = float(grid_feats['adjacent_grid_hamming_mean'])

        initial_solution = np.vstack([x, y]).T
        try:
            hypervolume = float(calculate_hypervolume(initial_solution, ref_point))
        except Exception:
            hypervolume = 0.0

        pareto_idx = find_global_pareto_points_simple(objective_values)
        pareto_decisions = np.array(decision_vars)[pareto_idx] if len(pareto_idx) > 0 else np.array([])
        if len(pareto_decisions) > 1:
            n = len(pareto_decisions)
            total = 0
            pairs = 0
            for i in range(n):
                for j in range(i + 1, n):
                    total += calculate_java_hamming_distance(pareto_decisions[i], pareto_decisions[j])
                    pairs += 1
            average_hamming_distance = float(total / pairs) if pairs > 0 else 0.0
        else:
            average_hamming_distance = 0.0

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Results/Output-draw')
    os.makedirs(output_dir, exist_ok=True)
    csv_file = f'{output_dir}/{mode}_statistics_reverse.csv' if reverse else f'{output_dir}/{mode}_statistics.csv'

    headers = [
        'Figure Number', 'Sampling Method', 'Sample Size', 'Dataset Name', 'Mode', 'Random Seed',
        'Unique ID', 'Sample Type',
        'adjacent grid hamming mean',
        'average hamming distance',
        'avg discrete descent cone',
        'connected components (pareto)',
        'connected components',
        'dominance change frequency',
        'grid density variance',
        'hypervolume',
        "kendall's tau correlation coefficient",
        'level count',
        'local efficient points count',
        'local-to-global ratio',
        'max dominance count',
        'median dominance count',
        'objective function value entropy',
        'objective function value variance',
        'pareto grid ratio',
        'pearson correlation coefficient',
        'slope of linear regression fit',
        'spearman correlation coefficient',
        'pareto dominated solution point ratio'
    ]

    row = [
        fig_num, sampling_method, sample_size, dataset_name, mode, random_seed, unique_id, sample_type,
        adjacent_grid_hamming_mean,
        average_hamming_distance,
        avg_discrete_descent_cone,
        connected_components_pareto,
        connected_components,
        dominance_change_frequency_stat,
        grid_density_var,
        hypervolume,
        kendalltau_corr,
        level_count,
        local_efficient_points_count,
        local_to_global_ratio,
        max_dominance_count,
        median_dominance_count,
        objective_function_value_entropy,
        objective_function_value_variance,
        pareto_grid_ratio,
        pearson_corr,
        slope,
        spearman_corr,
        pareto_ratio
    ]

    file_exists = os.path.isfile(csv_file)
    try:
        with open(csv_file, 'a', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)
    except Exception as e:
        print(f"Error writing to CSV file: {e}")


def find_center_point(r0_points, random_seed=42, center_type='mean'):
    if center_type == 'random':
        np.random.seed(random_seed)
        return r0_points[np.random.choice(len(r0_points))]
    elif center_type == 'mean':
        return np.mean(r0_points, axis=0)
    elif center_type == 'left_bottom_pareto':
        pareto_points = []
        for point in r0_points:
            if is_pareto_optimal_minimize(point, r0_points):
                pareto_points.append(point)
        pareto_points = np.array(pareto_points)
        if len(pareto_points) > 0:
            left_bottom_index = np.argmin(pareto_points[:, 0] + pareto_points[:, 1])
            return pareto_points[left_bottom_index]
        else:
            np.random.seed(random_seed)
            return r0_points[np.random.choice(len(r0_points))]
    else:
        raise ValueError("center_type must be 'random', 'mean' or 'left_bottom_pareto'.")


def plot_figure1(random_seed=42, center_type='random', sampling_method='', sample_size='',
                 dataset_name='', mode='', unique_id='', sample_type='', sampled_dict=None,
                 reverse=False, dag=None, num_valid_tasks=None):
    if not sampled_dict:
        print("plot_figure1: sampled_dict is empty")
        return
    sampled_data = list(sampled_dict.keys())
    r0_points = np.array(list(sampled_dict.values()))
    center_point = find_center_point(r0_points, random_seed, center_type)
    all_points = np.vstack([r0_points, center_point])

    r_pareto_count = {f'r_{i}': {'pareto': 0, 'non_pareto': 0} for i in range(8)}
    total_points_count = {'pareto': 0, 'non_pareto': 0}
    for pt in r0_points:
        is_p = is_pareto_optimal_minimize(pt, all_points)
        key = 'pareto' if is_p else 'non_pareto'
        total_points_count[key] += 1
        angle = np.arctan2(pt[1] - center_point[1], pt[0] - center_point[0]) % (2 * np.pi)
        for i in range(8):
            if i * np.pi / 4 <= angle < (i + 1) * np.pi / 4:
                r_pareto_count[f'r_{i}'][key] += 1
                break

    x = r0_points[:, 0]
    y = r0_points[:, 1]

    dominance_change_frequency = calculate_dominance_change_frequency(sampled_dict, random_seed=random_seed)

    decision_vars = list(sampled_dict.keys())
    global_idx = find_global_pareto_points_simple(np.array(list(sampled_dict.values())))
    pareto_set = [decision_vars[i] for i in global_idx]
    try:
        gdx = calculate_java_hamming_distance
    except Exception:
        pass

    print_and_save_statistics(sampled_data, sampled_dict, r_pareto_count, total_points_count,
                              1, x, y, random_seed, sampling_method, sample_size, dataset_name,
                              mode, unique_id, sample_type, dominance_change_frequency, reverse, dag, num_valid_tasks)


def plot_figure2(random_seed=42, center_type='random', sampling_method='', sample_size='',
                 dataset_name='', mode='', unique_id='', sample_type='', sampled_dict=None,
                 reverse=False, dag=None, num_valid_tasks=None):
    if not sampled_dict:
        print("plot_figure2: sampled_dict is empty")
        return
    sampled_data = list(sampled_dict.keys())
    objective_values = np.array(list(sampled_dict.values()))
    g1 = objective_values[:, 0]
    g2 = objective_values[:, 1]
    r0_points = np.column_stack((g1, g2))
    center_point = find_center_point(r0_points, random_seed, center_type)
    all_points = np.vstack([r0_points, center_point])

    r_pareto_count = {f'r_{i}': {'pareto': 0, 'non_pareto': 0} for i in range(8)}
    total_points_count = {'pareto': 0, 'non_pareto': 0}
    for pt in r0_points:
        is_p = is_pareto_optimal_minimize(pt, all_points)
        key = 'pareto' if is_p else 'non_pareto'
        total_points_count[key] += 1
        angle = np.arctan2(pt[1] - center_point[1], pt[0] - center_point[0]) % (2 * np.pi)
        for i in range(8):
            if i * np.pi / 4 <= angle < (i + 1) * np.pi / 4:
                r_pareto_count[f'r_{i}'][key] += 1
                break

    x = r0_points[:, 0]
    y = r0_points[:, 1]

    dominance_change_frequency = calculate_dominance_change_frequency(sampled_dict, random_seed=random_seed)

    print_and_save_statistics(sampled_data, sampled_dict, r_pareto_count, total_points_count,
                              2, x, y, random_seed, sampling_method, sample_size, dataset_name,
                              mode, unique_id, sample_type, dominance_change_frequency, reverse, dag, num_valid_tasks)