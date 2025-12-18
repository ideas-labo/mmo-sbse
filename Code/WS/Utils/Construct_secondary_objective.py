from collections import defaultdict

import numpy as np
import random
from scipy import spatial
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def generate_fa(configurations, ft, fa_construction, minimize, file_path, unique_elements_per_column, t, t_max,
                random_seed=42, age_info=None, novelty_archive=None, k=15):
    np.random.seed(random_seed)
    ft_normalized = normalize(np.array(ft))
    fa = []
    if fa_construction == 'age':
        if age_info is None:
            raise ValueError("Age information is required for age_maximization mode.")
        for i in range(len(ft)):
            age = age_info[i]
            fa_val = -age
            fa.append(fa_val)
    elif fa_construction == 'novelty':
        if novelty_archive is None:
            raise ValueError("Novelty archive is required for novelty_maximization mode.")

        for conf in configurations:
            novelty = calculate_novelty(conf, novelty_archive, k)
            fa_val = -novelty
            fa.append(fa_val)
    else:
        local_optima_indices = calculate_Proportion_of_local_optimal(configurations, ft, minimize) if fa_construction == 'penalty' else []
        penalty_features = find_penalty_features(configurations, ft, local_optima_indices) if fa_construction == 'penalty' else []
        for i in range(len(ft)):
            if fa_construction == 'gaussian':
                noise = np.random.normal(0, 1)
                if 'SS-C' in file_path or 'SS-B' in file_path:
                    fa_val = ft_normalized[i] - noise
                else:
                    fa_val = ft_normalized[i] + noise
                fa.append(fa_val)
            elif fa_construction == 'penalty':
                lambda_param = 0.1
                penalty = 0
                for feature, (cost, penalty_value) in penalty_features.items():
                    if has_feature(configurations[i], feature):
                        penalty += lambda_param * penalty_value * cost
                if i in local_optima_indices:
                    if 'SS-C' in file_path or 'SS-B' in file_path:
                        fa_val = ft_normalized[i] - penalty
                    else:
                        fa_val = ft_normalized[i] + penalty
                else:
                    fa_val = ft_normalized[i]
                fa.append(fa_val)
            elif fa_construction == 'diversity':
                theta = (1 - ((t - 1) / t_max))
                d_grid = calculate_d_grid(np.array(configurations[i]), np.array(configurations), theta,unique_elements_per_column)
                fa.append(d_grid)
            elif fa_construction =='reciprocal':
                if ft[i] != 0:
                    reciprocal_val = 1 / ft[i]
                    fa_val = -reciprocal_val
                else:
                    last_non_zero_ft = next((ft[j] for j in range(i - 1, -1, -1) if ft[j] != 0), None)
                    if last_non_zero_ft is not None:
                        reciprocal_val = 1 / last_non_zero_ft
                        fa_val = -reciprocal_val
                    else:
                        fa_val = -np.inf
                fa.append(fa_val)
    fa_normalized = normalize(np.array(fa))
    return ft_normalized, fa_normalized

def update_novelty_archive(solutions, archive, max_archive_size=1000):
    new_entries = []
    for solution in solutions:
        conf = tuple(solution.variables)
        if not any(np.isinf(obj) for obj in solution.objectives):
            novelty = calculate_novelty(conf, archive)
            new_entries.append((conf, novelty))

    archive.extend(new_entries)
    seen = set()
    unique_archive = []
    for conf, _ in archive:
        if conf not in seen:
            seen.add(conf)
            unique_archive.append((conf, archive[archive.index((conf, _))][1]))

    if len(unique_archive) > max_archive_size:
        unique_archive.sort(key=lambda x: x[1])
        unique_archive = unique_archive[-max_archive_size:]

    archive[:] = unique_archive


def calculate_novelty(conf, archive, k=15, max_candidates=500):
    if not archive:
        return 0

    archive_confs = [item[0] for item in archive] if isinstance(archive[0], tuple) else archive

    if not archive_confs:
        return 0

    dimension = len(archive_confs[0])
    hamming_index = ExactHammingIndex(dimension)
    hamming_index.add(archive_confs)

    _, distances = hamming_index.search(conf, k, max_candidates)

    if not distances:
        return 0
    elif len(distances) < k:
        return np.mean(distances)
    else:
        return np.mean(sorted(distances)[:k])


def calculate_d_grid(x, population, theta, unique_elements_per_column):
    if len(population)==0:
        return 0.0

    N = len(population)
    D = len(x)

    column_maps = []
    for col_vals in unique_elements_per_column:
        val_to_idx = {val: idx + 1 for idx, val in enumerate(col_vals)}
        column_maps.append(val_to_idx)

    x_grid = []
    for col_idx in range(D):
        val = x[col_idx]
        x_grid.append(column_maps[col_idx].get(val, 0))
    x_grid = np.array(x_grid, dtype=int)

    G_grid = []
    for config in population:
        grid_coord = []
        for col_idx in range(D):
            val = config[col_idx]
            grid_coord.append(column_maps[col_idx].get(val, 0))
        G_grid.append(grid_coord)
    G_grid = np.array(G_grid, dtype=int)

    hamming_index = ExactHammingIndex(D)
    hamming_index.add(G_grid)
    _, distances = hamming_index.search(x_grid, N)

    distances = np.array(distances)
    min_non_zero_dist = np.min(distances[distances > 0]) if np.any(distances > 0) else D
    distances[distances == 0] = D * 2

    S = theta * min_non_zero_dist if min_non_zero_dist != 0 else 1.0
    S_int = int(np.ceil(S)) if not np.isinf(S) else D

    mask = distances <= S_int
    neighborhood_size = np.sum(mask)
    distance_sum = np.sum(distances[mask]) if np.any(mask) else 0.0
    Diver = - (neighborhood_size + distance_sum)

    return Diver


class ExactHammingIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.data = []
        self._position_index = defaultdict(list)

    def add(self, vectors):
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.int32)
        start_idx = len(self.data)
        self.data.extend(vectors.tolist())
        for i in range(start_idx, len(self.data)):
            for pos in range(self.dimension):
                val = self.data[i][pos]
                self._position_index[(pos, val)].append(i)

    def search(self, query, k, max_candidates=1000):
        if not self.data:
            return [], []
        query = np.array(query, dtype=np.int32).flatten()
        candidate_counts = defaultdict(int)
        for pos in range(self.dimension):
            val = query[pos]
            for idx in self._position_index.get((pos, val), []):
                candidate_counts[idx] += 1
        sorted_candidates = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)
        candidate_indices = [idx for idx, _ in sorted_candidates[:max_candidates]]
        distances = []
        query_arr = np.array(query)
        for idx in candidate_indices:
            target = np.array(self.data[idx])
            distances.append(np.sum(query_arr != target))
        sorted_indices = np.argsort(distances)[:k]
        nearest_indices = [candidate_indices[i] for i in sorted_indices]
        nearest_distances = [distances[i] for i in sorted_indices]
        return ([tuple(self.data[i]) for i in nearest_indices], nearest_distances)

    def _get_adaptive_k(self):
        if self.dimension <= 200:    return 20
        elif self.dimension <= 500: return 15
        else:                       return 10


def calculate_Proportion_of_local_optimal(configurations, ft, minimize=True):
    if not configurations:
        return []

    dimension = len(configurations[0])
    hamming_index = ExactHammingIndex(dimension)
    hamming_index.add(configurations)

    config_ft_map = {tuple(config): val for config, val in zip(configurations, ft)}
    local_optima_indices = []

    k_neighbors = hamming_index._get_adaptive_k()

    for idx, config in enumerate(configurations):
        config_tuple = tuple(config)
        current_ft = config_ft_map[config_tuple]

        neighbors, _ = hamming_index.search(config, k_neighbors)

        is_local_optimum = True
        for n in neighbors:
            if n in config_ft_map and not is_better(current_ft, config_ft_map[n], minimize):
                is_local_optimum = False
                break

        if is_local_optimum:
            local_optima_indices.append(idx)

    return local_optima_indices


def is_better(x_value, y_value, minimize):
    if minimize:
        return x_value <= y_value
    else:
        return x_value >= y_value

def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.full_like(data, 0.5)
    return (data - min_val) / (max_val - min_val)


def has_feature(config, feature):
    feature_index, value = feature
    return config[feature_index] == value


def find_penalty_features(configurations, ft, local_optima_indices, penalty_threshold=0.1):
    num_configs = len(configurations)
    if num_configs == 0:
        return {}

    num_features = len(configurations[0])
    penalty_features = {}
    ft_normalized = normalize(np.array(ft))

    for feature_index in range(num_features):
        feature_values = [config[feature_index] for config in configurations]
        unique_values, counts = np.unique(feature_values, return_counts=True)

        for value in unique_values:
            total_ft = 0
            num_instances = 0
            for i in range(num_configs):
                if configurations[i][feature_index] == value:
                    total_ft += ft_normalized[i]
                    num_instances += 1

            avg_ft = total_ft / num_instances if num_instances > 0 else 0

            I_i = any(configurations[i][feature_index] == value
                      for i in local_optima_indices)

            current_penalty = penalty_features.get((feature_index, value), (0, 0))[1]

            util_i = I_i * (avg_ft / (1 + current_penalty))

            if util_i > penalty_threshold:
                penalty_increment = util_i * 1
                new_penalty = current_penalty + penalty_increment
                penalty_features[(feature_index, value)] = (avg_ft, new_penalty)

    return penalty_features