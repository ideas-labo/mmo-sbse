from scipy import spatial
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append('../..')
sys.path.append('../')
from Code.SCT.Utils.Construct_secondary_objective import generate_fa
from Code.SCT.Utils.Data_utils import SolutionHolder

MAXIMIZATION_DATASETS = ['storm_wc','storm_rs','storm_sol', 'dnn_dsr', 'dnn_coffee', 'dnn_adiac','x264','trimesh','dnn_coffee','dnn_dsr','x264']

class GotoFailedLabelException(Exception):
    pass

def get_objective_score_with_similarity(dict_search, best_solution):
    best_tuple = tuple(best_solution)
    if best_tuple in dict_search:
        return dict_search[best_tuple], best_solution.copy()

    keys = list(dict_search.keys())
    if not keys:
        raise ValueError("The dict_search cannot be empty.")

    scaler = MinMaxScaler()
    scaler.fit(keys)
    vectors = scaler.transform(keys)
    query_vect = scaler.transform([best_solution])[0]

    kdtree = spatial.KDTree(vectors)
    _, idx = kdtree.query(query_vect, k=1)

    nearest_key = keys[idx]
    nearest_value = dict_search[nearest_key]

    return nearest_value, list(nearest_key)

def get_objective_score_similarly(best_solution, dict_search, model_name, global_pareto_front, budget1):
    if model_name == "real":
        tmp_result, x = get_objective_score_with_similarity(dict_search, best_solution)

        is_dominated = False
        removes = []

        for i, pf in enumerate(global_pareto_front):
            if dominates(pf, tmp_result):
                is_dominated = True
                break
            if dominates(tmp_result, pf):
                removes.append(i)

        for i in reversed(removes):
            del global_pareto_front[i]

        return tmp_result

def dominates(solution1, solution2):
    if solution1[0] <= solution2[0] and solution1[1] < solution2[1]:
        return True
    elif solution1[0] < solution2[0] and solution1[1] <= solution2[1]:
        return True
    return False

def update_objectives(seed, population, mode, scaler_ft, scaler_fa, minimize, filename, unique_elements_per_column, dict_search, t, t_max, age_info=None, novelty_archive=None,w=1):
    for solution in population:
        if not hasattr(solution, 'original_objectives'):
            try:
                decision = [solution.variables[i] for i in range(len(solution.variables))]
                solution.original_objectives, _ = get_objective_score_similarly(decision, dict_search, "real", [], 0)
            except GotoFailedLabelException:
                raise
    if mode == 'ft_fa':
        for solution in population:
            ft, fa = solution.original_objectives
            solution.objectives = [ft, fa]

    elif mode in ['gaussian_fa', 'penalty_fa', 'reciprocal_fa', 'age_maximization_fa', 'novelty_maximization_fa', 'diversity_fa']:
        ft_list = [ind.original_objectives[0] for ind in population]
        configurations = [tuple(ind.variables) for ind in population]
        mode_construction = mode.split('_')[0]
        ft_normalized, fa_normalized = generate_fa(configurations, ft_list, mode_construction, minimize, filename, unique_elements_per_column, t, t_max, random_seed=seed, age_info=age_info, novelty_archive=novelty_archive, k=len(population) // 2)

        for i, solution in enumerate(population):
            solution_holder = SolutionHolder(0, [], [ft_normalized[i], fa_normalized[i]])
            solution_holder.normalized_ft = ft_normalized[i]
            solution_holder.normalized_fa = fa_normalized[i]
            solution_holder.transformed_g1 = ft_normalized[i] + w*fa_normalized[i]
            solution_holder.transformed_g2 = ft_normalized[i] - w*fa_normalized[i]
            solution.objectives = [solution_holder.transformed_g1, solution_holder.transformed_g2]

    elif mode == 'g1_g2':
        all_ft = [ind.original_objectives[0] for ind in population]
        all_fa = [ind.original_objectives[1] for ind in population]
        min_ft = min(all_ft)
        max_ft = max(all_ft)
        min_fa = min(all_fa)
        max_fa = max(all_fa)

        for solution in population:
            ft, fa = solution.original_objectives
            solution_holder = SolutionHolder(0, [], [ft, fa])
            if max_ft - min_ft == 0:
                solution_holder.normalized_ft = 0
            else:
                solution_holder.normalized_ft = (ft - min_ft) / (max_ft - min_ft)
            if max_fa - min_fa == 0:
                solution_holder.normalized_fa = 0
            else:
                solution_holder.normalized_fa = (fa - min_fa) / (max_fa - min_fa)
            solution_holder.transformed_g1 = solution_holder.normalized_ft + w*solution_holder.normalized_fa
            solution_holder.transformed_g2 = solution_holder.normalized_ft - w*solution_holder.normalized_fa
            solution.objectives = [solution_holder.transformed_g1, solution_holder.transformed_g2]

    return population