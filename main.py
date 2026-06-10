from src.DataLoader import ProjectDataLoader
from src.evaluation import evaluate_dataset, print_results_table
from src.plotting import plot_time_metrics, plot_memory_metrics, plot_combined_dual_axis, plot_2d_embeddings
import json
import numpy as np
import argparse


def run_pipeline(num_runs=10, fast_dev=False):
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

def save_results_to_json(results, baselines, filename="metrics_results.json"):
    """Recursively converts numpy types and saves results cleanly to JSON."""

    def serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            # JSON keys must be strings, convert dimension ints to strings
            return {str(k): serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(v) for v in obj]
        return obj

    data = {
        "baselines": serialize(baselines),
        "results": serialize(results)
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"\n[SUCCESS] All metrics and arrays exported to local file: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=10)

    args = parser.parse_args()
    results, baselines = run_pipeline(num_runs=args.num_runs, fast_dev=False)
    print_results_table(results, baselines)

    # Save a hard copy of results before plotting
    save_results_to_json(results, baselines)

    # Import the newly added line chart functions
    from src.plotting import plot_classification_metrics, plot_unsupervised_metrics

    # 1. Plot the new line charts across ALL evaluated dimensions (2, 3, 4, 5)
    plot_classification_metrics(results, baselines)
    plot_unsupervised_metrics(results)

    # 2. Keep the original bar/embedding plots cleanly mapped to d=2 slice only
    d2 = {ds: {2: data[2]} for ds, data in results.items()}
    plot_time_metrics(d2)
    plot_memory_metrics(d2)
    plot_combined_dual_axis(d2)
    plot_2d_embeddings(d2)