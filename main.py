import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
import umap

from DataLoader import ProjectDataLoader

# Set plotting style for clean academic visuals
sns.set_theme(style="whitegrid")


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


def plot_computational_metrics(all_results):
    """Generates bar charts comparing execution time and peak memory footprint."""
    for ds_name, models_data in all_results.items():
        models = list(models_data.keys())
        times = [data["time"] for data in models_data.values()]
        mems = [data["memory_mb"] for data in models_data.values()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Computational Evaluation: {ds_name}", fontsize=14, fontweight='bold')

        # Execution Time Barplot
        sns.barplot(x=models, y=times, ax=ax1, palette="viridis")
        ax1.set_title("Execution Time")
        ax1.set_ylabel("Seconds")
        ax1.set_xlabel("Model")

        # Peak Memory Barplot
        sns.barplot(x=models, y=mems, ax=ax2, palette="magma")
        ax2.set_title("Peak Memory Usage")
        ax2.set_ylabel("Megabytes (MB)")
        ax2.set_xlabel("Model")

        plt.tight_layout()
        plt.savefig(f"{ds_name.lower().replace(' ', '_')}_computational_metrics.png", dpi=300)
        plt.show()


def plot_2d_embeddings(all_results):
    """Generates a grid of 2D scatter plots for visual manifold inspection."""
    for ds_name, models_data in all_results.items():
        num_models = len(models_data)
        if num_models == 0: continue

        fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4.5))
        fig.suptitle(f"2D Projections: {ds_name}", fontsize=16, fontweight='bold')

        # Ensure axes is iterable even if only 1 model ran
        if num_models == 1:
            axes = [axes]

        for ax, (model_name, data) in zip(axes, models_data.items()):
            X_2d = data["reduced_data"]
            labels = data["labels"]

            # Note: PBMC 3k labels default to 0 in your current DataLoader unless updated
            is_monochromatic = len(np.unique(labels)) <= 1

            scatter = ax.scatter(
                X_2d[:, 0], X_2d[:, 1],
                c=labels,
                cmap="tab10" if not is_monochromatic else None,
                alpha=0.6,
                s=2
            )
            ax.set_title(f"{model_name}", fontsize=12)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

            # Add legend for datasets with explicit targets (like Fashion-MNIST)
            if not is_monochromatic and ds_name == "Fashion-MNIST":
                legend = ax.legend(*scatter.legend_elements(), title="Classes", loc="upper right", fontsize='small')
                ax.add_artist(legend)

        plt.tight_layout()
        plt.savefig(f"{ds_name.lower().replace(' ', '_')}_2d_embeddings.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    pipeline_results = run_pipeline()

    plot_computational_metrics(pipeline_results)
    plot_2d_embeddings(pipeline_results)