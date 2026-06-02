from DataLoader import ProjectDataLoader
from evaluation import evaluate_dataset, print_results_table
from plotting import plot_time_metrics, plot_memory_metrics, plot_combined_dual_axis, plot_2d_embeddings


def run_pipeline(num_runs=3, fast_dev=False):
    datasets = _load_datasets(fast_dev)
    all_results, baselines = {}, {}

    for ds_name, (X, y) in datasets.items():
        print(f"\n{'=' * 60}\n  Dataset: {ds_name}\n{'=' * 60}")
        baseline, dim_data = evaluate_dataset(ds_name, X, y, num_runs=num_runs, fast_dev=fast_dev)
        baselines[ds_name]   = baseline
        all_results[ds_name] = dim_data

    return all_results, baselines


def _load_datasets(fast_dev):
    if fast_dev:
        print("[FAST DEV MODE] Bypassing data loader.\n")
        import numpy as np
        return {
            "Fashion-MNIST": (np.random.rand(500, 784).astype("float32"), np.random.randint(0, 10, 500)),
            "PBMC 3k":       (np.random.rand(300, 100).astype("float32"), np.random.randint(0, 8,  300)),
        }
    DL = ProjectDataLoader()
    return {
        "Fashion-MNIST": DL.load_fashion_mnist(),
        "PBMC 3k":       DL.load_pbmc_3k(),
    }


if __name__ == "__main__":
    results, baselines = run_pipeline(fast_dev=False)
    print_results_table(results, baselines)

    # Plotting (d=2 only)
    d2 = {ds: {2: data[2]} for ds, data in results.items()}
    plot_time_metrics(d2)
    plot_memory_metrics(d2)
    plot_combined_dual_axis(d2)
    plot_2d_embeddings(d2)