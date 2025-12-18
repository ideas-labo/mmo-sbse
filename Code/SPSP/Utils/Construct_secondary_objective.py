from collections import defaultdict

import numpy as np
import random
from scipy import spatial
from scipy.spatial import KDTree
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def generate_fa(configurations, ft, fa_construction, minimize, unique_elements_per_column, t, t_max,
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
                fa_val = ft_normalized[i] + noise
                fa.append(fa_val)
            elif fa_construction == 'penalty':
                lambda_param = 0.1
                penalty = 0
                for feature, (cost, penalty_value) in penalty_features.items():
                    if has_feature(configurations[i], feature):
                        penalty += lambda_param * penalty_value * cost
                if i in local_optima_indices:
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
        conf = _round_vector(solution.variables)
        if not any(np.isinf(obj) for obj in solution.objectives):
            exists = any(_round_vector(item[0]) == conf for item in archive)
            if not exists:
                novelty = calculate_novelty(conf, archive)
                new_entries.append((conf, novelty))

    archive.extend(new_entries)

    if len(archive) > max_archive_size:
        archive.sort(key=lambda x: x[1])
        archive = archive[-max_archive_size:]

import hnswlib
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

from collections import defaultdict
import numpy as np
from scipy.spatial import KDTree

class ExactEuclideanIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.data = []
        self.kd_tree = None
        self._needs_rebuild = True

    def add(self, vectors):
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float64)
        self.data.extend(vectors.tolist())
        self._needs_rebuild = True

    def _rebuild_tree(self):
        if self._needs_rebuild and len(self.data) > 0:
            self.kd_tree = KDTree(np.array(self.data, dtype=np.float64))
            self._needs_rebuild = False

    def search(self, query, k):
        if len(self.data) == 0:
            return [], []
        self._rebuild_tree()
        query = np.array(query, dtype=np.float64).reshape(1, -1)
        distances, indices = self.kd_tree.query(query, k=k)
        nearest_configs = [self.data[i] for i in indices[0]]
        return nearest_configs, distances[0].tolist()

    def _get_adaptive_k(self):
        if self.dimension <= 200:
            return 15
        elif self.dimension <= 500:
            return 10
        else:
            return 5

def calculate_novelty(conf, archive, k=15):
    if not archive:
        return 0.0
    if len(archive)<k:
        k=len(archive)
    dimension = len(archive[0][0]) if isinstance(archive[0], tuple) else len(archive[0])
    nn_index = ExactEuclideanIndex(dimension)

    archive_data = []
    for item in archive:
        if isinstance(item, tuple):
            archive_data.append(np.array(item[0], dtype=np.float64))
        else:
            archive_data.append(np.array(item, dtype=np.float64))
    nn_index.add(archive_data)

    query = np.array(conf, dtype=np.float64)
    _, distances = nn_index.search(query, k)

    if not distances:
        return 0.0
    return np.mean(distances)

def calculate_Proportion_of_local_optimal(configurations, ft, minimize=True):
    if not configurations:
        return []

    config_map = {tuple(conf): val for conf, val in zip(configurations, ft)}

    dimension = len(configurations[0])
    nn_index = ExactEuclideanIndex(dimension)
    nn_index.add(configurations)

    k_neighbors = nn_index._get_adaptive_k()
    if len(configurations)<k_neighbors:
        k_neighbors=len(configurations)
    local_optima_indices = []

    for idx, conf in enumerate(configurations):
        current_ft = config_map.get(tuple(conf), float('inf'))

        neighbors, _ = nn_index.search(conf, k_neighbors)

        is_optimum = True
        for neighbor in neighbors:
            neighbor_ft = config_map.get(tuple(neighbor), float('inf'))
            if not is_better(current_ft, neighbor_ft, minimize):
                is_optimum = False
                break

        if is_optimum:
            local_optima_indices.append(idx)

    return local_optima_indices

def calculate_d_grid_numba(x, population, theta, unique_elements_per_column):
    if len(population) == 0:
        return 0.0

    x_arr = np.array(x, dtype=np.float64)
    pop_arr = np.array(population, dtype=np.float64)

    kd_tree = KDTree(pop_arr)
    distances, _ = kd_tree.query(x_arr.reshape(1, -1), k=len(pop_arr))
    distances = distances[0]

    min_non_zero = np.min(distances[distances > 0]) if np.any(distances > 0) else len(x)
    S_float = theta * min_non_zero
    S_int = int(np.ceil(S_float)) if min_non_zero != 0 else len(x)

    mask = (distances > 0) & (distances <= S_int)
    neighborhood_size = np.sum(mask)
    distance_sum = np.sum(distances[mask]) if np.any(mask) else 0.0

    return -(neighborhood_size + distance_sum)

def _round_vector(vector, precision=6):
    return tuple(round(float(x), precision) for x in vector)

import numpy as np
from numba import njit

@njit
def build_grid_mapping(unique_elements_per_column):
    value_to_index = []
    for col_vals in unique_elements_per_column:
        mapping = np.zeros(len(col_vals))
        for idx, val in enumerate(col_vals):
            mapping[idx] = round(val, 6)
        value_to_index.append(mapping)
    return value_to_index


@njit(fastmath=True)
def calculate_euclidean_distance_numba(x_grid, pop_grid):
    N, D = pop_grid.shape
    distances = np.zeros(N, dtype=np.float64)

    for i in range(N):
        dist_sq = 0.0
        for j in range(D):
            diff = x_grid[j] - pop_grid[i, j]
            dist_sq += diff * diff
        distances[i] = np.sqrt(dist_sq)

    return distances

@njit
def calculate_d_grid_numba(x, population, theta, unique_elements_per_column):
    if len(population) == 0:
        return 0.0

    value_to_index = build_grid_mapping(unique_elements_per_column)
    D = len(x)
    N = len(population)

    x_grid = np.zeros(D, dtype=np.int32)
    for j in range(D):
        val = round(x[j], 6)
        found = -1
        for k in range(len(value_to_index[j])):
            if abs(value_to_index[j][k] - val) < 1e-6:
                found = k + 1
                break
        x_grid[j] = found if found != -1 else 0

    pop_grid = np.zeros((N, D), dtype=np.int32)
    for i in range(N):
        for j in range(D):
            val = round(population[i][j], 6)
            found = -1
            for k in range(len(value_to_index[j])):
                if abs(value_to_index[j][k] - val) < 1e-6:
                    found = k + 1
                    break
            pop_grid[i, j] = found if found != -1 else 0

    distances = calculate_euclidean_distance_numba(x_grid, pop_grid)

    min_non_zero = np.min(distances[distances > 0]) if np.any(distances > 0) else D
    S_float = theta * min_non_zero
    S_int = int(np.ceil(S_float)) if min_non_zero != 0 else D

    neighborhood_size = 0
    distance_sum = 0
    for d in distances:
        if 0 < d <= S_int:
            neighborhood_size += 1
            distance_sum += d

    return -(neighborhood_size + distance_sum)


def calculate_d_grid(x, population, theta, unique_elements_per_column):
    x_arr = np.array(x, dtype=np.float64)
    pop_arr = np.array(population, dtype=np.float64)
    ue_list = [np.array(col, dtype=np.float64) for col in unique_elements_per_column]

    return calculate_d_grid_numba(x_arr, pop_arr, theta, ue_list)

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
    return round(float(config[feature_index]), 6) == round(float(value), 6)


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