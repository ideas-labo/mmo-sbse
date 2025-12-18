import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler

MAXIMIZATION_DATASETS_FT = ['storm_wc','storm_rs','storm_sol', 'dnn_dsr', 'dnn_coffee', 'dnn_adiac','dnn_sa','trimesh','dnn_coffee','dnn_dsr','x264']
MAXIMIZATION_DATASETS_FA = []

class SolutionHolder:
    def __init__(self, id, decisions, objective):
        self.id = id
        self.decision = decisions
        self.objective = objective
        self.normalized_ft = None
        self.normalized_fa = None
        self.transformed_g1 = None
        self.transformed_g2 = None

    def normalize_and_transform(self, scaler_ft=None, scaler_fa=None, min_ft=None, max_ft=None, min_fa=None,
                                max_fa=None):
        ft, fa = self.objective
        if scaler_ft and scaler_fa:
            self.normalized_ft = scaler_ft.transform([[ft]])[0][0]
            self.normalized_fa = scaler_fa.transform([[fa]])[0][0]
        elif min_ft is not None and max_ft is not None and min_fa is not None and max_fa is not None:
            if max_ft - min_ft == 0:
                self.normalized_ft = 0
            else:
                self.normalized_ft = (ft - min_ft) / (max_ft - min_ft)
            if max_fa - min_fa == 0:
                self.normalized_fa = 0
            else:
                self.normalized_fa = (fa - min_fa) / (max_fa - min_fa)
        self.transformed_g1 = self.normalized_ft + self.normalized_fa
        self.transformed_g2 = self.normalized_ft - self.normalized_fa

class FileData:
    def __init__(self, name, training_set, testing_set, all_set, independent_set, features, dict_search, scaler_ft,
                 scaler_fa, reverse=False):
        self.name = name
        self.training_set = training_set
        self.testing_set = testing_set
        self.all_set = all_set
        self.independent_set = independent_set
        self.features = features
        self.dict_search = dict_search
        self.scaler_ft = scaler_ft
        self.scaler_fa = scaler_fa
        self.reverse = reverse

def get_data(filename, initial_size=5, seed=1, reverse=False):
    random.seed(seed)
    np.random.seed(seed)
    try:
        with open(filename, 'r') as f:
            header = f.readline().strip().split(',')
        dep_indices = [i for i, col in enumerate(header) if "<$" in col]
        indep_indices = [i for i in range(len(header)) if i not in dep_indices]
        depcolumns = [header[i] for i in dep_indices]
        indepcolumns = [header[i] for i in indep_indices]

        if len(depcolumns) == 2:
            depcolumns = ['ft', 'fa']

        tmp_sortindepcolumns = []
        data = []
        with open(filename, 'r') as f:
            next(f)
            for line in f:
                try:
                    row = [float(x) for x in line.strip().split(',')]
                    if len(row) == len(header):
                        data.append(row)
                except ValueError:
                    continue

        data = np.array(data)

        indep_data = data[:, indep_indices]
        for i in range(indep_data.shape[1]):
            tmp_sortindepcolumns.append(sorted(list(set(indep_data[:, i]))))
        for i in range(len(tmp_sortindepcolumns)):
            for j in range(len(tmp_sortindepcolumns[i])):
                tmp_sortindepcolumns[i][j] = float(tmp_sortindepcolumns[i][j])

        content = []
        objectives = []

        is_maximization_dataset_ft = any(dataset in filename for dataset in MAXIMIZATION_DATASETS_FT)
        is_maximization_dataset_fa = any(dataset in filename for dataset in MAXIMIZATION_DATASETS_FA)

        for c in range(data.shape[0]):
            ft_index = dep_indices[0]
            fa_index = dep_indices[1] if len(dep_indices) > 1 else None
            ft_value = data[c, ft_index]
            fa_value = data[c, fa_index] if fa_index is not None else None

            if is_maximization_dataset_ft:
                ft_value = -ft_value

            if is_maximization_dataset_fa and fa_value is not None:
                fa_value = -fa_value

            if reverse and fa_value is not None:
                ft_value, fa_value = fa_value, ft_value
            objective = [ft_value, fa_value] if fa_value is not None else [ft_value]
            content.append(SolutionHolder(
                c,
                indep_data[c].tolist(),
                objective
            ))
            objectives.append(objective)

        ft_values = [obj[0] for obj in objectives]
        fa_values = [obj[1] for obj in objectives if len(obj) > 1]

        scaler_ft = MinMaxScaler()
        scaler_fa = MinMaxScaler()
        scaler_ft.fit(np.array(ft_values).reshape(-1, 1))
        if fa_values:
            scaler_fa.fit(np.array(fa_values).reshape(-1, 1))

        for solution in content:
            solution.normalize_and_transform(scaler_ft=scaler_ft, scaler_fa=scaler_fa if fa_values else None)

        dict_search = dict(zip(
            [tuple(i.decision) for i in content],
            [i.objective for i in content]
        ))

        random.seed(seed)
        random.shuffle(content)

        indexes = range(len(content))
        train_indexes, test_indexes = indexes[:initial_size], indexes[initial_size:]
        assert (len(train_indexes) + len(test_indexes) == len(indexes)), "Dataset split error"

        train_set = [content[i] for i in train_indexes]
        test_set = [content[i] for i in test_indexes]

        file = FileData(
            filename, train_set, test_set, content, tmp_sortindepcolumns, indepcolumns, dict_search, scaler_ft, scaler_fa if fa_values else None,
            reverse
        )
        return file
    except Exception as e:
        print(f"Error reading file: {e}")
        return None