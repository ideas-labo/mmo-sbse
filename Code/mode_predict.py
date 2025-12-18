import os
import warnings
import random
from sklearn.metrics import ndcg_score
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
import lightgbm as lgb
import torch

def load_and_preprocess_data(file_path, test_dataset):
    data = pd.read_csv(file_path)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Data size after removing NaNs and infinities: {len(data)}")

    if 'Dataset Name' not in data.columns:
        raise ValueError("Missing required column 'Dataset Name' in data")
    if 'ft_rank' not in data.columns:
        raise ValueError("Missing required column 'ft_rank' in data")

    if test_dataset not in data['Dataset Name'].unique():
        raise ValueError(f"Test dataset {test_dataset} is not in available datasets")

    train_mask = data['Dataset Name'] != test_dataset
    train_data = data[train_mask].copy()
    test_data = data[~train_mask].copy()

    print(f"\n=== Leave-one-dataset split ===")
    print(f"Number of training datasets: {len(data['Dataset Name'].unique()) - 1}")
    print(f"Test dataset: {test_dataset}")
    print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")

    return train_data, test_data

def train_and_evaluate_direct_rank(train_data, test_data, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
        random.seed(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("\n=== Direct ranking mode: predict ranking per sampling method + compute NDCG ===")

    exclude_cols = [
        'Dataset Name', 'Random Seed', 'mode', 'Sampling Method',
        'Optimal_Best_Pareto_Ratio', 'Optimal_Pareto_Ratios_Mean',
        'Best_Pareto_Ratio', 'Pareto_Ratios_Mean',
        'Percent_Diff_Best_P', 'Percent_Diff_P_Mean',
        'ft_rank', 'time_rank', 'budget_rank',
        'ft', 'time', 'budget',
        'GDx_MMO', 'GDx_PMO','Sample Size'
    ]
    exclude_cols = [col for col in exclude_cols if col in train_data.columns]

    feature_cols = [col for col in train_data.columns if col not in exclude_cols]
    X_train_raw = train_data[feature_cols].copy()
    X_test_raw = test_data[feature_cols].copy()

    print(f"\n=== Feature extraction ===")
    print(f"Original number of features: {len(feature_cols)}")
    print(f"Feature list: {feature_cols}")

    X_train_filtered = X_train_raw.copy()
    X_test_filtered = X_test_raw[X_train_filtered.columns]
    removed_features = []

    final_feature_cols = X_train_filtered.columns.tolist()
    print(f"Number of features after filtering (no filtering applied): {len(final_feature_cols)}")

    categorical_features = []
    if 'mode_encoded' in final_feature_cols:
        categorical_features.append('mode_encoded')
    if 'sampling_method_encoded' in final_feature_cols:
        categorical_features.append('sampling_method_encoded')

    print(f"Identified categorical features: {categorical_features}")

    def safe_preprocess(X, train_stats=None):
        X = X.copy()

        if train_stats is None:
            num_cols = [col for col in X.columns
                        if col not in categorical_features and
                        pd.api.types.is_numeric_dtype(X[col])]

            train_stats = {
                'mean': X[num_cols].mean() if num_cols else pd.Series(),
                'categorical_modes': {},
                'num_cols': num_cols,
                'cat_cols': categorical_features
            }

            for col in categorical_features:
                if col in X.columns:
                    mode_val = X[col].mode()
                    train_stats['categorical_modes'][col] = mode_val[0] if len(mode_val) > 0 else 0

        if train_stats['num_cols']:
            X[train_stats['num_cols']] = X[train_stats['num_cols']].fillna(train_stats['mean'])

        for col, mode_val in train_stats['categorical_modes'].items():
            if col in X.columns:
                X[col] = X[col].fillna(mode_val)

        remaining_nan = X.isnull().sum().sum()
        if remaining_nan > 0:
            print(f"  Warning: {remaining_nan} missing values remain after preprocessing")
            X = X.fillna(0)

        return X, train_stats

    print("\n=== Feature preprocessing ===")
    X_train_processed, train_stats = safe_preprocess(X_train_filtered)
    X_test_processed, _ = safe_preprocess(X_test_filtered, train_stats)

    print(f"Train shape: {X_train_processed.shape}")
    print(f"Test shape: {X_test_processed.shape}")
    print(f"Train missing values: {X_train_processed.isnull().sum().sum()}")
    print(f"Test missing values: {X_test_processed.isnull().sum().sum()}")

    y_train = train_data['ft_rank'].values
    y_test = test_data['ft_rank'].values

    max_rank = int(max(np.max(y_train), np.max(y_test)))
    print(f"\n=== Training LambdaMART model ===")
    print(f"Detected max rank: {max_rank}")

    lambdamart_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, max_rank],
        'max_position': max_rank,
        'lambdarank_truncation_level': max_rank,
        'boosting_type': 'gbdt',
        'num_leaves': 30,
        'learning_rate': 0.05,
        'min_data_in_leaf': max(5, int(len(X_train_processed) / (max_rank * 2))),
        'feature_fraction': 0.7,
        'bagging_fraction': 0.9,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'seed': random_state,
        'verbosity': -1
    }

    train_groups = train_data.groupby('Dataset Name').size().values
    print(f"Number of groups in train set: {len(train_groups)}, group sizes: {train_groups}")

    lgb_train = lgb.Dataset(
        X_train_processed,
        label=y_train,
        group=train_groups,
        categorical_feature=categorical_features if categorical_features else 'auto'
    )

    rank_model = lgb.train(
        lambdamart_params,
        lgb_train,
        num_boost_round=100,
        callbacks=[
            lgb.reset_parameter(learning_rate=lambda epoch: lambdamart_params['learning_rate'] * (0.99 ** epoch)),
            lgb.log_evaluation(period=10)
        ]
    )

    print("\n=== Testing: predict by sampling method ===")
    grouped_test_data = test_data.groupby('Sampling Method')
    all_sampling_results = []

    for sampling_method, group_indices in grouped_test_data.groups.items():
        print(f"\n--- Processing sampling method: {sampling_method} ---")
        X_group = X_test_processed.loc[group_indices]
        group_samples = test_data.loc[group_indices]

        if len(X_group) == 0:
            print(f"  Warning: sampling method {sampling_method} has no test samples, skipping")
            continue

        group_scores = rank_model.predict(X_group, num_iteration=rank_model.best_iteration)

        dataset_results = []
        y_true_ndcg_all = []
        y_pred_ndcg_all = []

        for dataset in group_samples['Dataset Name'].unique():
            dataset_mask = group_samples['Dataset Name'] == dataset
            dataset_samples = group_samples[dataset_mask]
            dataset_scores = group_scores[dataset_mask]

            if len(dataset_samples) == 0:
                continue

            sorted_indices = np.argsort(dataset_scores)
            sorted_modes = dataset_samples['mode'].values[sorted_indices]
            mode_ranks = {mode: i + 1 for i, mode in enumerate(sorted_modes)}

            modes = sorted(set(dataset_samples['mode'].values))
            true_ranks = {mode: dataset_samples[dataset_samples['mode'] == mode]['ft_rank'].iloc[0] for mode in modes}
            pred_ranks = {mode: mode_ranks.get(mode, len(modes) + 1) for mode in modes}

            modes_by_pred = sorted(modes, key=lambda x: pred_ranks[x])
            modes_by_true = sorted(modes, key=lambda x: true_ranks[x])

            true_rank_list = [{'mode': mode, 'rank': true_ranks[mode]} for mode in modes_by_true]
            pred_rank_list = [{'mode': mode, 'rank': pred_ranks[mode]} for mode in modes_by_pred]

            max_rank_dataset = max(true_ranks.values()) if true_ranks else 1
            y_true_rel = np.array([max_rank_dataset - true_ranks[mode] + 1 for mode in modes]).reshape(1, -1)
            y_pred_rel = np.array([max_rank_dataset - pred_ranks[mode] + 1 for mode in modes]).reshape(1, -1)
            y_true_ndcg_all.append(y_true_rel)
            y_pred_ndcg_all.append(y_pred_rel)

            correct = sum(1 for mode in modes if pred_ranks[mode] == true_ranks[mode])
            accuracy = correct / len(modes) if modes else 0

            dataset_results.append({
                'dataset': dataset,
                'accuracy': accuracy,
                'total_modes': len(modes),
                'correct': correct,
                'true_ranking': true_rank_list,
                'pred_ranking': pred_rank_list
            })

            print(f"  Dataset {dataset}:")
            print(f"    {'Mode':<15} | {'True Rank':<10} | {'Pred Rank':<10}")
            print(f"    {'-' * 40}")
            for mode in modes_by_pred:
                print(f"    {mode:<15} | {true_ranks[mode]:<10} | {pred_ranks[mode]:<10}")
            print(f"    Accuracy: {accuracy:.2f} ({correct}/{len(modes)})")

        ndcg = ndcg_at1 = ndcg_at3 = np.nan
        if y_true_ndcg_all and y_pred_ndcg_all:
            y_true_ndcg_stack = np.vstack(y_true_ndcg_all)
            y_pred_ndcg_stack = np.vstack(y_pred_ndcg_all)
            ndcg = ndcg_score(y_true_ndcg_stack, y_pred_ndcg_stack)
            ndcg_at1 = ndcg_score(y_true_ndcg_stack, y_pred_ndcg_stack, k=1)
            ndcg_at3 = ndcg_score(y_true_ndcg_stack, y_pred_ndcg_stack, k=min(3, y_true_ndcg_stack.shape[1]))

        sampling_result = {
            'sampling_method': sampling_method,
            'ndcg': ndcg,
            'ndcg_at1': ndcg_at1,
            'ndcg_at3': ndcg_at3,
            'random_state': random_state,
            'dataset_results': dataset_results
        }
        all_sampling_results.append(sampling_result)

        print(f"\n  Global metrics for sampling method {sampling_method}:")
        print(f"    NDCG: {ndcg:.4f}")
        print(f"    NDCG@1: {ndcg_at1:.4f}")
        print(f"    NDCG@3: {ndcg_at3:.4f}")

    return all_sampling_results

def check_feature_consistency(X_train, X_test):
    train_features = set(X_train.columns)
    test_features = set(X_test.columns)

    missing_in_test = train_features - test_features
    missing_in_train = test_features - train_features

    if missing_in_test:
        raise ValueError(f"Test set is missing these train features: {missing_in_test}")
    if missing_in_train:
        raise ValueError(f"Test set contains features not in train set: {missing_in_train}")

    if list(X_train.columns) != list(X_test.columns):
        print("Warning: train and test feature order mismatch. Adjusting test feature order.")
        X_test = X_test[X_train.columns]

    print("✅ Train/test feature consistency check passed")
    return X_test

def check_data_leakage(X_train, X_test, y_train, y_test):
    leak_detected = False

    train_samples = set(X_train.index)
    test_samples = set(X_test.index)
    overlap = train_samples & test_samples
    if overlap:
        print(f"Warning: {len(overlap)} overlapping samples detected")
        leak_detected = True

    for col in X_train.columns:
        if X_train[col].equals(X_test[col]):
            print(f"Warning: feature '{col}' is identical in train and test")
            leak_detected = True

    if set(y_train) == set(y_test):
        print("Warning: target variable identical in train and test")
        leak_detected = True

    if not leak_detected:
        print("✅ No data leakage detected")

    return not leak_detected

def main():
    input_folder = '../Results/Predict-raw-data/ProcessedData'
    result_folder = '../Results/Predict-raw-data/Model_performance'
    num_runs = 10

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    all_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.csv')]
    print(f"Found {len(all_files)} CSV files: {all_files}")

    for fname in all_files:
        file_path = os.path.join(input_folder, fname)

        suffix = fname.split('_')[-1].rsplit('.', 1)[0]
        result_subfolder = os.path.join(result_folder, suffix)
        if not os.path.exists(result_subfolder):
            os.makedirs(result_subfolder)

        all_results = []

        print(f"\nProcessing file: {file_path}")
        for run in range(num_runs):
            random_state = run
            print(f"\n{'=' * 50}")
            print(f"Starting run {run + 1}/{num_runs} (random seed: {random_state})")
            print(f"{'=' * 50}")

            data = pd.read_csv(file_path)
            unique_datasets = data['Dataset Name'].unique()
            print(f"Found {len(unique_datasets)} datasets: {list(unique_datasets)}")

            for test_dataset in unique_datasets:
                print(f"\n{'=' * 50}")
                print(f"Current test dataset: {test_dataset}")
                print(f"{'=' * 50}")

                train_data, test_data = load_and_preprocess_data(
                    file_path,
                    test_dataset=test_dataset
                )

                sampling_results = train_and_evaluate_direct_rank(
                    train_data,
                    test_data,
                    random_state=random_state
                )

                for res in sampling_results:
                    res['test_dataset'] = test_dataset
                    all_results.append(res)

        if all_results:
            import copy
            results_for_df = copy.deepcopy(all_results)

            for res in results_for_df:
                if 'dataset_results' in res:
                    for dr in res['dataset_results']:
                        if 'accuracy' in dr:
                            dr.pop('accuracy', None)
                    res['dataset_results_str'] = str(res['dataset_results'])
                    del res['dataset_results']

            all_results_df = pd.DataFrame(results_for_df)

            if 'run' in all_results_df.columns:
                all_results_df = all_results_df.drop(columns=['run'])

            if 'sampling_method' not in all_results_df.columns:
                all_results_df['sampling_method'] = 'all_methods'

            numeric_cols = ['ndcg', 'ndcg_at1', 'ndcg_at3']

            raw_result_path = os.path.join(result_subfolder, f'all_runs_raw_results.csv')
            all_results_df.to_csv(raw_result_path, index=False, encoding='utf-8')
            print(f"\nAll raw sampling method results for the file saved to: {raw_result_path}")

            avg_results = all_results_df.groupby(['sampling_method', 'test_dataset'])[numeric_cols].agg(
                ['mean', 'std']).reset_index()

            overall_avg = all_results_df.groupby(['sampling_method'])[numeric_cols].agg(
                ['mean', 'std']).reset_index()
            overall_avg['test_dataset'] = 'overall_average'

            final_avg_results = pd.concat([avg_results, overall_avg], ignore_index=True, sort=False)

            avg_result_path = os.path.join(result_subfolder, f'average_results.csv')
            final_avg_results.to_csv(avg_result_path, index=False, encoding='utf-8')
            print(f"Average results per sampling method saved to: {avg_result_path}")

            try:
                print("\n=== Overall average metrics per sampling method ===")
                print(overall_avg.round(4))
            except Exception:
                pass

if __name__ == "__main__":
    main()