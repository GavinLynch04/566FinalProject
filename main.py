import time
import tracemalloc
import numpy as np
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
import umap

from DataLoader import ProjectDataLoader
from plotting import plot_computational_metrics, plot_2d_embeddings


def get_models():
    """Returns freshly instantiated models targeting 2D embeddings."""
    return {
        "PCA": PCA(n_components=2),
        "t-SNE": TSNE(n_components=2, random_state=42, n_jobs=-1),
        "Isomap": Isomap(n_components=2, n_jobs=-1),
        "UMAP": umap.UMAP(n_components=2, random_state=42)
    }


def run_pipeline(subsample_size=3000):
    DL = ProjectDataLoader()

    # Define datasets to evaluate
    datasets = {
        "Fashion-MNIST": DL.load_fashion_mnist(),
        "PBMC 3k": DL.load_pbmc_3k()
    }

    all_results = {}

    for ds_name, (X, y) in datasets.items():
        print(f"\n========== Processing Dataset: {ds_name} ==========")

        # Subsampling safety valve for computationally heavy models (Isomap/t-SNE)
        if subsample_size and X.shape[0] > subsample_size:
            print(f"Subsampling dataset from {X.shape[0]} to {subsample_size} for performance safety.")
            indices = np.random.choice(X.shape[0], subsample_size, replace=False)
            X_eval, y_eval = X[indices], y[indices]
        else:
            X_eval, y_eval = X, y

        all_results[ds_name] = {}
        models = get_models()

        for model_name, model_instance in models.items():
            print(f"Running {model_name}...")

            # Reset and start memory tracking
            tracemalloc.start()
            start_time = time.perf_counter()

            try:
                # Fit and transform data
                X_reduced = model_instance.fit_transform(X_eval)

                end_time = time.perf_counter()
                _, peak_memory = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                # Save metrics and embeddings
                all_results[ds_name][model_name] = {
                    "time": end_time - start_time,
                    "memory_mb": peak_memory / (1024 * 1024),  # Convert bytes to MB
                    "reduced_data": X_reduced,
                    "labels": y_eval
                }
                print(
                    f"Finished {model_name}: Time = {end_time - start_time:.2f}s | Peak Mem = {peak_memory / (1024 * 1024):.2f} MB")

            except Exception as e:
                tracemalloc.stop()
                print(f"Failed to run {model_name} on {ds_name}: {e}")

    return all_results

if __name__ == "__main__":
    pipeline_results = run_pipeline()

    plot_computational_metrics(pipeline_results)
    plot_2d_embeddings(pipeline_results)