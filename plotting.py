import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set plotting style for clean academic visuals
sns.set_theme(style="whitegrid")

# figure directory
figure_directory = "figures/"
os.makedirs(figure_directory, exist_ok=True)

def plot_computational_metrics(all_results):
    """Generates bar charts comparing execution time and peak memory footprint."""
    for ds_name, models_data in all_results.items():
        models = list(models_data.keys())
        times = [data["time"] for data in models_data.values()]
        mems = [data["memory_mb"] for data in models_data.values()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Computational Evaluation: {ds_name}", fontsize=14, fontweight='bold')

        # Execution Time Barplot
        sns.barplot(x=models, y=times, ax=ax1, hue=models, palette="viridis", legend=False)
        ax1.set_title("Execution Time")
        ax1.set_ylabel("Seconds")
        ax1.set_xlabel("Model")

        # Peak Memory Barplot
        sns.barplot(x=models, y=mems, ax=ax2, hue=models, palette="magma", legend=False)
        ax2.set_title("Peak Memory Usage")
        ax2.set_ylabel("Megabytes (MB)")
        ax2.set_xlabel("Model")

        plt.tight_layout()
        plt.savefig(f"{figure_directory}{ds_name.lower().replace(' ', '_')}_computational_metrics.png", dpi=300)
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
        plt.savefig(f"{figure_directory}{ds_name.lower().replace(' ', '_')}_2d_embeddings.png", dpi=300)
        plt.show()