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


def train_and_evaluate_direct_rank(train_data, test_data, random_state=None,
                                   lambdamart_params=None, num_boost_round=500):
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
        'GDx_MMO', 'GDx_PMO', 'Sample Size'
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

    def convert_ranks_to_relevance(data):
        relevance_data = data.copy()

        dataset_max_ranks = {}
        for dataset in data['Dataset Name'].unique():
            dataset_ranks = data[data['Dataset Name'] == dataset]['ft_rank']
            dataset_max_ranks[dataset] = int(dataset_ranks.max())

        relevance_list = []
        for dataset in data['Dataset Name'].unique():
            dataset_mask = data['Dataset Name'] == dataset
            max_rank = dataset_max_ranks[dataset]
            relevance = max_rank - data.loc[dataset_mask, 'ft_rank'] + 1
            relevance_list.append(relevance)

        return pd.concat(relevance_list, axis=0).values

    y_train = convert_ranks_to_relevance(train_data)
    y_test = convert_ranks_to_relevance(test_data)

    print(f"\n=== Label conversion statistics ===")
    print(f"Training set: Original rank range = {train_data['ft_rank'].min()}-{train_data['ft_rank'].max()}, "
          f"Converted relevance range = {y_train.min()}-{y_train.max()}")
    print(f"Test set: Original rank range = {test_data['ft_rank'].min()}-{test_data['ft_rank'].max()}, "
          f"Converted relevance range = {y_test.min()}-{y_test.max()}")

    if y_train.min() <= 0:
        print(f"Warning: Training set relevance scores contain non-positive values! Min value = {y_train.min()}")
        offset = -y_train.min() + 1
        y_train = y_train + offset
        y_test = y_test + offset
        print(f"Adjusted training set relevance range: {y_train.min()}-{y_train.max()}")

    max_relevance = int(max(np.max(y_train), np.max(y_test)))
    print(f"Detected maximum relevance score: {max_relevance}")

    default_lambdamart_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, max_relevance],
        'max_position': max_relevance,
        'lambdarank_truncation_level': max_relevance,
        'boosting_type': 'gbdt',
        'num_leaves': 30,
        'learning_rate': 0.05,
        'min_data_in_leaf': 10,
        'bagging_fraction': 0.9,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'seed': random_state,
        'verbosity': -1,
        'feature_fraction': 0.7,
        'min_data': 1,
        'min_child_samples': 1,
        'min_split_gain': 0.0,
        'max_delta_step': 0.1,
    }

    if lambdamart_params is not None:
        final_params = default_lambdamart_params.copy()

        for key, value in lambdamart_params.items():
            if key == 'ndcg_eval_at':
                if isinstance(value, list):
                    if max_relevance not in value:
                        value = list(value) + [max_relevance]
                    final_params[key] = value
                else:
                    final_params[key] = value
            elif key == 'max_position' or key == 'lambdarank_truncation_level':
                final_params[key] = value
            elif key == 'seed':
                final_params[key] = random_state
            else:
                final_params[key] = value

        print(f"Using custom LambdaMART parameters:")
        for key, value in final_params.items():
            if key in ['seed', 'max_position', 'lambdarank_truncation_level', 'ndcg_eval_at']:
                print(f"  {key}: {value}")
    else:
        final_params = default_lambdamart_params
        print(f"Using default LambdaMART parameters")

    if 'max_position' not in final_params or final_params['max_position'] < max_relevance:
        final_params['max_position'] = max_relevance
    if 'lambdarank_truncation_level' not in final_params or final_params['lambdarank_truncation_level'] < max_relevance:
        final_params['lambdarank_truncation_level'] = max_relevance

    train_groups = train_data.groupby('Dataset Name').size().values
    print(f"Number of groups in training set: {len(train_groups)}, Group sizes: {train_groups}")

    lgb_train = lgb.Dataset(
        X_train_processed,
        label=y_train,
        group=train_groups,
        categorical_feature=categorical_features if categorical_features else 'auto'
    )

    print(f"\n=== Training LambdaMART model ===")
    print(f"Boosting rounds: {num_boost_round}")

    rank_model = lgb.train(
        final_params,
        lgb_train,
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.log_evaluation(period=10)
        ]
    )

    print("\n=== Test: Predict by sampling method ===")
    grouped_test_data = test_data.groupby('Sampling Method')
    all_sampling_results = []

    for sampling_method, group_indices in grouped_test_data.groups.items():
        print(f"\n--- Processing sampling method: {sampling_method} ---")
        X_group = X_test_processed.loc[group_indices]
        group_samples = test_data.loc[group_indices]

        if len(X_group) == 0:
            print(f"  Warning: No test samples for sampling method {sampling_method}, skipping")
            continue

        group_scores = rank_model.predict(X_group, num_iteration=rank_model.best_iteration)

        print(f"  Prediction score statistics: Min={group_scores.min():.4f}, "
              f"Max={group_scores.max():.4f}, Mean={group_scores.mean():.4f}")

        dataset_results = []
        y_true_ndcg_all = []
        y_pred_scores_all = []

        for dataset in group_samples['Dataset Name'].unique():
            dataset_mask = group_samples['Dataset Name'] == dataset
            dataset_samples = group_samples[dataset_mask]
            dataset_scores = group_scores[dataset_mask]

            if len(dataset_samples) == 0:
                continue

            sorted_indices = np.argsort(-dataset_scores)
            sorted_modes = dataset_samples['mode'].values[sorted_indices]
            mode_ranks = {mode: i + 1 for i, mode in enumerate(sorted_modes)}

            modes = sorted(set(dataset_samples['mode'].values))

            true_ranks = {mode: dataset_samples[dataset_samples['mode'] == mode]['ft_rank'].iloc[0] for mode in modes}

            max_rank_dataset = max(true_ranks.values()) if true_ranks else 1
            true_relevance_list = []
            true_rank_list = []

            for mode in modes:
                true_rank = true_ranks[mode]
                true_relevance = max_rank_dataset - true_rank + 1
                true_relevance_list.append(true_relevance)
                true_rank_list.append(true_rank)

            true_relevance_array = np.array(true_relevance_list).reshape(1, -1)

            pred_scores_list = []
            pred_ranks_list = []

            for mode in modes:
                mode_mask = dataset_samples['mode'] == mode
                if np.any(mode_mask):
                    pred_score = dataset_scores[mode_mask][0]
                    pred_scores_list.append(pred_score)
                    pred_ranks_list.append(mode_ranks.get(mode, len(modes) + 1))
                else:
                    pred_scores_list.append(np.min(dataset_scores) if len(dataset_scores) > 0 else 0)
                    pred_ranks_list.append(len(modes) + 1)

            pred_scores_array = np.array(pred_scores_list).reshape(1, -1)

            y_true_ndcg_all.append(true_relevance_array)
            y_pred_scores_all.append(pred_scores_array)

            try:
                dataset_ndcg = ndcg_score(true_relevance_array, pred_scores_array)
                dataset_ndcg_at1 = ndcg_score(true_relevance_array, pred_scores_array, k=1)
                dataset_ndcg_at3 = ndcg_score(true_relevance_array, pred_scores_array, k=min(3, len(modes)))
            except Exception as e:
                print(f"    Error calculating dataset NDCG: {e}")
                dataset_ndcg = dataset_ndcg_at1 = dataset_ndcg_at3 = np.nan

            modes_by_pred = sorted(modes, key=lambda x: pred_ranks_list[modes.index(x)])
            true_rank_display = [{'mode': mode, 'rank': true_ranks[mode]} for mode in modes_by_pred]
            pred_rank_display = [{'mode': mode, 'rank': pred_ranks_list[modes.index(mode)]} for mode in modes_by_pred]

            correct = sum(1 for mode in modes if pred_ranks_list[modes.index(mode)] == true_ranks[mode])
            accuracy = correct / len(modes) if modes else 0

            dataset_results.append({
                'dataset': dataset,
                'accuracy': accuracy,
                'total_modes': len(modes),
                'correct': correct,
                'true_ranking': true_rank_display,
                'pred_ranking': pred_rank_display,
                'dataset_ndcg': dataset_ndcg,
                'dataset_ndcg_at1': dataset_ndcg_at1,
                'dataset_ndcg_at3': dataset_ndcg_at3
            })

            print(f"  Dataset {dataset}:")
            print(f"    True relevance: {true_relevance_list}")
            print(f"    Prediction scores: {[f'{s:.4f}' for s in pred_scores_list]}")
            print(f"    Mode | True Rank | Pred Rank | Pred Score")
            print(f"    {'-' * 45}")
            for i, mode in enumerate(modes_by_pred):
                true_rank = true_ranks[mode]
                pred_rank = pred_ranks_list[modes.index(mode)]
                pred_score = pred_scores_list[modes.index(mode)]
                print(f"    {mode:<10} | {true_rank:<8} | {pred_rank:<8} | {pred_score:.4f}")
            print(f"    Accuracy: {accuracy:.2f} ({correct}/{len(modes)})")
            print(f"    Dataset NDCG: {dataset_ndcg:.4f}, NDCG@1: {dataset_ndcg_at1:.4f}, NDCG@3: {dataset_ndcg_at3:.4f}")

        ndcg = ndcg_at1 = ndcg_at3 = np.nan
        if y_true_ndcg_all and y_pred_scores_all:
            try:
                y_true_ndcg_stack = np.vstack(y_true_ndcg_all)
                y_pred_scores_stack = np.vstack(y_pred_scores_all)

                ndcg = ndcg_score(y_true_ndcg_stack, y_pred_scores_stack)
                ndcg_at1 = ndcg_score(y_true_ndcg_stack, y_pred_scores_stack, k=1)
                ndcg_at3 = ndcg_score(y_true_ndcg_stack, y_pred_scores_stack,
                                      k=min(3, y_true_ndcg_stack.shape[1]))

                print(f"\n  Overall NDCG statistics:")
                print(f"    Number of samples: {len(y_true_ndcg_all)} datasets")
                print(f"    Average modes per dataset: {y_true_ndcg_stack.shape[1]}")
                print(f"    Prediction score range: {y_pred_scores_stack.min():.4f} - {y_pred_scores_stack.max():.4f}")

            except Exception as e:
                print(f"  Error calculating overall NDCG: {e}")

        sampling_result = {
            'sampling_method': sampling_method,
            'ndcg': ndcg,
            'ndcg_at1': ndcg_at1,
            'ndcg_at3': ndcg_at3,
            'random_state': random_state,
            'num_boost_round': num_boost_round,
            'lambdamart_params': final_params,
            'dataset_results': dataset_results
        }
        all_sampling_results.append(sampling_result)

        print(f"\n  Overall metrics for sampling method {sampling_method}:")
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

    dataset_params_config = {
        'nas': {
            'lambdamart_params': {
                'num_leaves': 50,
                'learning_rate': 0.01,
                'min_data_in_leaf': 15,
                'bagging_fraction': 0.75,
                'lambda_l1': 0.1,
                'lambda_l2': 0.05,
                'feature_fraction': 1,
            },
            'num_boost_round': 1000
        },
        'splt': {
            'lambdamart_params': {
                'num_leaves': 30,
                'learning_rate': 0.01,
                'min_data_in_leaf': 15,
                'bagging_fraction': 0.9,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'feature_fraction': 0.7,
            },
            'num_boost_round': 300
        },
        'ws': {
            'lambdamart_params': {
                'num_leaves': 30,
                'learning_rate': 0.05,
                'min_data_in_leaf': 15,
                'bagging_fraction': 0.9,
                'lambda_l1': 0.1,
                'lambda_l2': 0.2,
                'feature_fraction': 0.7,
            },
            'num_boost_round': 500
        },
        'spsp': {
            'lambdamart_params': {
                'num_leaves': 30,
                'learning_rate': 0.05,
                'min_data_in_leaf': 15,
                'bagging_fraction': 0.9,
                'lambda_l1': 0.1,
                'lambda_l2': 0.2,
                'feature_fraction': 0.7,
            },
            'num_boost_round': 500
        },
        'sct': {
            'lambdamart_params': {
                'num_leaves': 150,
                'learning_rate': 0.05,
                'min_data_in_leaf': 10,
                'bagging_fraction': 0.7,
                'lambda_l1': 0.1,
                'lambda_l2': 1,
                'feature_fraction': 1,
            },
            'num_boost_round': 1000
        },
        'wsc': {
            'lambdamart_params': {
                'num_leaves': 30,
                'learning_rate': 0.05,
                'min_data_in_leaf': 10,
                'bagging_fraction': 0.7,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'feature_fraction': 1,
            },
            'num_boost_round': 300
        },
        'sdp': {
            'lambdamart_params': {
                'num_leaves': 30,
                'learning_rate': 0.05,
                'min_data_in_leaf': 10,
                'bagging_fraction': 0.7,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'feature_fraction': 1,
            },
            'num_boost_round': 500
        },
        'see': {
            'lambdamart_params': {
                'num_leaves': 50,
                'learning_rate': 0.05,
                'min_data_in_leaf': 15,
                'bagging_fraction': 0.8,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'feature_fraction': 1,
            },
            'num_boost_round': 500
        },
        'default': {
            'lambdamart_params': None,
            'num_boost_round': 300
        }
    }

    for fname in all_files:
        file_path = os.path.join(input_folder, fname)

        suffix = fname.split('_')[-1].rsplit('.', 1)[0]
        result_subfolder = os.path.join(result_folder, suffix)
        if not os.path.exists(result_subfolder):
            os.makedirs(result_subfolder)

        all_results = []

        print(f"\nProcessing file: {file_path}")

        if suffix in dataset_params_config:
            params_config = dataset_params_config[suffix]
            print(f"Using custom parameters for suffix: {suffix}")
        else:
            params_config = dataset_params_config.get('default',
                                                      {'lambdamart_params': None, 'num_boost_round': 500})
            print(f"Using default parameters for suffix: {suffix}")

        lambdamart_params = params_config.get('lambdamart_params')
        num_boost_round = params_config.get('num_boost_round', 500)

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
                    random_state=random_state,
                    lambdamart_params=lambdamart_params,
                    num_boost_round=num_boost_round
                )

                for res in sampling_results:
                    res['test_dataset'] = test_dataset
                    res['file_suffix'] = suffix
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

                if 'lambdamart_params' in res:
                    res['lambdamart_params_str'] = str(res['lambdamart_params'])
                    del res['lambdamart_params']

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