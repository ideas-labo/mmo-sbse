import numpy as np
import sys
sys.path.append('../')

def generate_fa(configurations, ft, fa_construction, minimize, file_path, unique_elements_per_column, t, t_max,
                random_seed=42, age_info=None, novelty_archive=None, k=10):
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
        if len(novelty_archive) < k:
            raise ValueError(f"The size of novelty_archive ({len(novelty_archive)}) must be at least k ({k}).")

        for conf in configurations:
            novelty = calculate_novelty(conf, novelty_archive, k)
            fa_val = -novelty
            fa.append(fa_val)
    else:
        local_optima_indices = calculate_Proportion_of_local_optimal(configurations, ft, unique_elements_per_column,
                                                                     minimize) if fa_construction == 'penalty' else []
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
                d_grid = calculate_d_grid(np.array(configurations[i]), np.array(configurations), theta,
                                          unique_elements_per_column, random_seed)
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

def calculate_novelty(conf, archive, k=15):
    if not archive:
        return 0

    distances = []
    for stored_conf, _ in archive:
        distance = sum(1 for x, y in zip(conf, stored_conf) if x != y)
        distances.append(distance)

    if len(distances) < k:
        return np.mean(distances)
    else:
        return np.mean(sorted(distances)[:k])

def calculate_d_grid(x, population, theta, unique_elements_per_column, random_seed=42):
    np.random.seed(random_seed)
    N = len(population)
    D = len(x)

    column_maps = []
    for col_vals in unique_elements_per_column:
        val_to_idx = {val: idx + 1 for idx, val in enumerate(col_vals)}
        column_maps.append(val_to_idx)

    G = []
    for config in population:
        grid_coord = []
        for col_idx in range(D):
            val = config[col_idx]
            grid_coord.append(column_maps[col_idx].get(val, 0))
        G.append(grid_coord)
    G = np.array(G, dtype=int)

    x_grid = []
    for col_idx in range(D):
        val = x[col_idx]
        x_grid.append(column_maps[col_idx].get(val, 0))
    x_grid = np.array(x_grid, dtype=int)

    GD = np.zeros(N)
    for i in range(N):
        distance = np.sum(G[i] != x_grid)
        GD[i] = distance

    self_indices = np.where(np.all(G == x_grid, axis=1))[0]
    GD[self_indices] = np.inf

    if N == 0:
        S = 0.0
    else:
        min_distances = np.min(GD) if np.any(GD) else 0.0
        S = theta * np.max(min_distances) if min_distances != 0 else 1.0

    S_int = int(np.ceil(S))
    mask = GD <= S_int

    neighborhood_size = np.sum(mask)
    distance_sum = np.sum(GD[mask]) if np.any(mask) else 0.0
    Diver = - (neighborhood_size + distance_sum)

    return Diver

def find_k_nearest_neighbors(target_config, configurations, k=5):
    distances = []
    target_config = tuple(target_config)

    for config in configurations:
        config = tuple(config)
        if config == target_config:
            continue
        distance = sum(1 for x, y in zip(target_config, config) if x != y)
        distances.append((config, distance))

    distances.sort(key=lambda x: x[1])
    return [config for config, dist in distances[:k]]

def calculate_Proportion_of_local_optimal(configurations, ft, unique_elements_per_column, minimize=True):
    new_configurations = [tuple(config) for config in configurations]
    local_optima_indices = []
    config_ft_map = dict(zip(new_configurations, ft))
    population_size = len(configurations)

    dim = len(configurations[0]) if configurations else 0
    base_k = min(dim, 10)
    k = min(base_k, max(3, population_size // 5))

    for i, config in enumerate(new_configurations):
        neighbors = find_k_nearest_neighbors(config, new_configurations, k)
        is_local_optima = True

        for neighbor in neighbors:
            if neighbor in config_ft_map:
                if not is_better(config_ft_map[config], config_ft_map[neighbor], minimize):
                    is_local_optima = False
                    break

        if is_local_optima:
            local_optima_indices.append(i)

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
    num_features = len(configurations[0])
    penalty_features = {}
    ft_normalized = normalize(np.array(ft))

    penalty_values = {i: 0 for i in range(num_features)}

    for feature_index in range(num_features):
        feature_values = [config[feature_index] for config in configurations]
        unique_values, counts = np.unique(feature_values, return_counts=True)

        for value_index, value in enumerate(unique_values):
            total_ft = 0
            num_instances = 0
            for i in range(num_configs):
                config = configurations[i]
                if config[feature_index] == value:
                    total_ft += ft_normalized[i]
                    num_instances += 1

            if num_instances > 0:
                avg_ft = total_ft / num_instances
            else:
                avg_ft = 0

            try:
                I_i = sum([1 for i in local_optima_indices if configurations[i][feature_index] == value]) > 0
            except TypeError as e:
                print(f"TypeError 发生在 feature_index={feature_index}, value={value}: {e}")
                print(f"configurations 类型: {type(configurations)}, 长度: {len(configurations)}")
                print(f"local_optima_indices 内容: {local_optima_indices}")
                raise

            util_i = I_i * (avg_ft / (1 + penalty_values[feature_index]))

            if util_i > penalty_threshold:
                penalty_increment = util_i * 1
                penalty_values[feature_index] += penalty_increment

                cost = avg_ft
                penalty_value = penalty_values[feature_index]
                penalty_features[(feature_index, value)] = (cost, penalty_value)

    return penalty_features