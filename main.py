from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
import umap

from DataLoader import ProjectDataLoader
from plotting import plot_computational_metrics, plot_2d_embeddings
from helper import subsample_dataset, generate_mock_results, profile_model_performance


def get_models():
    """Returns freshly instantiated models targeting 2D embeddings."""
    return {
        "PCA": PCA(n_components=2),
        "t-SNE": TSNE(n_components=2, random_state=42, n_jobs=-1),
        "Isomap": Isomap(n_components=2, n_jobs=-1),
        "UMAP": umap.UMAP(n_components=2, random_state=42)
    }


def run_pipeline(subsample_size=3000, num_runs=10, fast_dev=False):
    dataset_names = ["Fashion-MNIST", "PBMC 3k"]
    all_results = {}

    # Bypass data loading entirely if in fast development mode
    if fast_dev:
        print("[FAST DEV MODE] Bypassing data loader. Simulating pipelines...")
        for ds_name in dataset_names:
            all_results[ds_name] = {}
            for model_name in get_models().keys():
                all_results[ds_name][model_name] = generate_mock_results(ds_name, model_name,
                                                                         num_samples=subsample_size)
        return all_results

    # Standard Production Pipeline execution
    DL = ProjectDataLoader()
    datasets = {
        "Fashion-MNIST": DL.load_fashion_mnist(),
        "PBMC 3k": DL.load_pbmc_3k()
    }

    for ds_name, (X, y) in datasets.items():
        print(f"\n========== Processing Dataset: {ds_name} ==========")
        X_eval, y_eval = subsample_dataset(X, y, subsample_size)
        all_results[ds_name] = {}

        for model_name, model_instance in get_models().items():
            print(f"Running {model_name} ({num_runs} runs)...")

            metrics = profile_model_performance(model_instance, X_eval, num_runs)
            if metrics:
                all_results[ds_name][model_name] = {**metrics, "labels": y_eval}
                print(
                    f"Finished {model_name}: Avg Time = {metrics['time']:.2f}s | Avg Mem = {metrics['memory_mb']:.2f} MB")
            else:
                print(f"Skipping results generation for {model_name} due to execution errors.")

    return all_results

if __name__ == "__main__":
    pipeline_results = run_pipeline(fast_dev=True)

    plot_computational_metrics(pipeline_results)
    plot_2d_embeddings(pipeline_results)