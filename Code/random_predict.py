import os
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
import warnings
warnings.filterwarnings("ignore")


def load_and_filter_data(file_path):
    data = pd.read_csv(file_path)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Data size after cleaning: {len(data)}")
    print(f"All datasets found: {data['Dataset Name'].nunique()}")
    print(f"All modes found: {data['mode'].nunique()}\n")
    return data


def calculate_random_baseline(data, random_seed):
    import random
    results = []

    np.random.seed(random_seed)
    random.seed(random_seed)

    unique_datasets = sorted(data['Dataset Name'].unique())

    for dataset in unique_datasets:
        dataset_data = data[data['Dataset Name'] == dataset]

        all_modes = sorted(dataset_data['mode'].unique())
        num_modes = len(all_modes)

        if num_modes < 2:
            continue

        true_ranks_dict = {}
        for mode in all_modes:
            mode_data = dataset_data[dataset_data['mode'] == mode]
            if not mode_data.empty:
                true_ranks_dict[mode] = mode_data['ft_rank'].iloc[0]
        true_ranks = np.array([true_ranks_dict[mode] for mode in all_modes])

        pred_ranks = np.random.permutation(range(1, num_modes + 1))

        max_rank = num_modes
        true_relevance = np.array([max_rank - r + 1 for r in true_ranks]).reshape(1, -1)
        pred_relevance = np.array([max_rank - r + 1 for r in pred_ranks]).reshape(1, -1)

        try:
            ndcg = ndcg_score(true_relevance, pred_relevance)
            ndcg_at1 = ndcg_score(true_relevance, pred_relevance, k=1)
            ndcg_at3 = ndcg_score(true_relevance, pred_relevance, k=min(3, max_rank))
        except:
            ndcg = ndcg_at1 = ndcg_at3 = np.nan

        results.append({
            'dataset': dataset,
            'num_modes': num_modes,
            'ndcg': ndcg,
            'ndcg_at1': ndcg_at1,
            'ndcg_at3': ndcg_at3,
            'modes': all_modes,
            'true_ranks': true_ranks,
            'pred_ranks': pred_ranks
        })

    return pd.DataFrame(results)


def main():
    input_folder = '../Results/Predict-raw-data/ProcessedData'
    num_runs = 10

    all_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.csv')]
    print(f"Found {len(all_files)} csv files: {all_files}")

    for fname in all_files:
        file_path = os.path.join(input_folder, fname)

        suffix = fname.split('_')[-1].rsplit('.', 1)[0]
        result_subfolder = os.path.join('../Results/Predict-raw-data/Model_performance', suffix)
        os.makedirs(result_subfolder, exist_ok=True)

        print(f"\n=== Loading data without filtering for file: {file_path} ===")
        data = load_and_filter_data(file_path)

        print(f"\n=== Running {num_runs} experiments with random baseline method for file: {fname} ===")

        individual_runs_list = []

        for seed in range(num_runs):
            print(f"Running experiment with seed: {seed}")
            run_results = calculate_random_baseline(data, random_seed=seed)

            run_results['run_number'] = seed
            individual_runs_list.append(run_results)

        if len(individual_runs_list) == 0:
            print(f"No valid runs produced results for file {fname}, skipping saving.")
            continue

        all_individual_results = pd.concat(individual_runs_list, ignore_index=True)

        individual_combined_file = os.path.join(result_subfolder, 'all_random_individual_runs.csv')

        individual_columns_to_save = ['dataset', 'num_modes', 'ndcg', 'ndcg_at1', 'ndcg_at3',
                                      'modes', 'true_ranks', 'pred_ranks', 'run_number']
        cols_to_write = [c for c in individual_columns_to_save if c in all_individual_results.columns]
        all_individual_results[cols_to_write].to_csv(individual_combined_file, index=False)
        print(f"All individual runs results saved to: {individual_combined_file}")

        numeric_cols = ['ndcg', 'ndcg_at1', 'ndcg_at3']

        avg_results_df = all_individual_results.groupby('dataset')[numeric_cols].mean().reset_index()
        std_results_df = all_individual_results.groupby('dataset')[numeric_cols].std().reset_index()

        avg_results_df.columns = ['dataset'] + [f'{col}_mean' for col in numeric_cols]
        std_results_df.columns = ['dataset'] + [f'{col}_std' for col in numeric_cols]

        final_results_df = pd.merge(avg_results_df, std_results_df, on='dataset')

        first_run_info = all_individual_results[all_individual_results['run_number'] == 0][
            ['dataset', 'modes', 'true_ranks']]
        final_results_df = final_results_df.merge(first_run_info, on='dataset', how='left')

        avg_result_file = os.path.join(result_subfolder, 'average_random_rank.csv')

        avg_columns_to_save = ['dataset', 'ndcg_mean', 'ndcg_std', 'ndcg_at1_mean', 'ndcg_at1_std',
                               'ndcg_at3_mean', 'ndcg_at3_std', 'modes', 'true_ranks']
        cols_to_write_avg = [c for c in avg_columns_to_save if c in final_results_df.columns]
        final_results_df[cols_to_write_avg].to_csv(avg_result_file, index=False)
        print(f"Average results with standard deviation across {num_runs} runs saved to: {avg_result_file}")

        print("\n" + "=" * 50)
        print(f"Average NDCG metrics across all datasets (over {num_runs} runs) for file {fname}:")
        print("=" * 50)
        overall_avg = final_results_df[[col for col in final_results_df.columns if '_mean' in col]].mean()
        print(overall_avg.round(4))

        print(f"\n=== Processing of file {fname} completed successfully ===\n")


if __name__ == "__main__":
    main()