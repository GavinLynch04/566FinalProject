import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

sns.set_theme(style="white")

figure_directory = "figures/"
os.makedirs(figure_directory, exist_ok=True)

# Color Map by model
COLOR_MAP = {
    "PCA": {"dark": "#3cb4dd", "light": "#93d6ec"},
    "t-SNE": {"dark": "#dd3c57", "light": "#ec93a2"},
    "Isomap": {"dark": "#7f3cdd", "light": "#b893ec"},
    "UMAP": {"dark": "#3cdd8c", "light": "#93ecbf"},
    "VAE": {"dark": "#dd9a3c", "light": "#ecc793"}
}


def get_model_colors(models, shade):
    """Fetches the 'dark' or 'light' color variant for a list of models."""
    return [COLOR_MAP.get(m, {"dark": "#333333", "light": "#aaaaaa"})[shade] for m in models]


def format_standard_axes(ax):
    """Applies a unified black axis, horizontal light grey grid, and despines top/right/left."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_color('black')
    ax.tick_params(axis='x', colors='black')

    ax.tick_params(axis='y', colors='black', left=False)

    ax.yaxis.grid(True, color='lightgray', linestyle='-')
    ax.xaxis.grid(False)

    ax.set_axisbelow(True)


def plot_time_metrics(all_results):
    """Bar charts for execution time with standard deviation."""
    num_ds = len(all_results)
    fig, axes = plt.subplots(1, num_ds, figsize=(6 * num_ds, 5), sharey=True)
    if num_ds == 1: axes = [axes]

    fig.suptitle("Execution Time Comparison", fontsize=15, fontweight='bold')

    for ax, (ds_name, models_data) in zip(axes, all_results.items()):
        models = list(models_data.keys())
        times = [data["time_mean"] for data in models_data.values()]
        stds = [data["time_std"] for data in models_data.values()]
        colors = get_model_colors(models, "dark")

        # Added saturation=1.0 to prevent Seaborn from washing out the custom hex codes
        sns.barplot(x=models, y=times, ax=ax, hue=models, palette=colors, legend=False,
                    edgecolor='black', linewidth=1, saturation=1.0)

        x_coords = np.arange(len(models))
        ax.errorbar(x=x_coords, y=times, yerr=stds, fmt='none', c='black', capsize=4, elinewidth=1.2)

        ax.set_title(ds_name, fontsize=13)
        if ax == axes[0]:
            ax.set_ylabel("Seconds")

        format_standard_axes(ax)

    plt.tight_layout()
    plt.savefig(f"{figure_directory}execution_time_combined.png", dpi=600)
    plt.show()


def plot_memory_metrics(all_results):
    """Bar charts for peak memory footprint with standard deviation."""
    num_ds = len(all_results)
    fig, axes = plt.subplots(1, num_ds, figsize=(6 * num_ds, 5), sharey=True)
    if num_ds == 1: axes = [axes]

    fig.suptitle("Peak Memory Footprint", fontsize=15, fontweight='bold')

    for ax, (ds_name, models_data) in zip(axes, all_results.items()):
        models = list(models_data.keys())
        mems = [data["memory_mean"] for data in models_data.values()]
        stds = [data["memory_std"] for data in models_data.values()]
        colors = get_model_colors(models, "dark")

        # Added saturation=1.0 to prevent Seaborn from washing out the custom hex codes
        sns.barplot(x=models, y=mems, ax=ax, hue=models, palette=colors, legend=False,
                    edgecolor='black', linewidth=1, saturation=1.0)

        x_coords = np.arange(len(models))
        ax.errorbar(x=x_coords, y=mems, yerr=stds, fmt='none', c='black', capsize=4, elinewidth=1.2)

        ax.set_title(ds_name, fontsize=13)
        if ax == axes[0]:
            ax.set_ylabel("Megabytes (MB)")

        format_standard_axes(ax)

    plt.tight_layout()
    plt.savefig(f"{figure_directory}memory_footprint_combined.png", dpi=600)
    plt.show()


def plot_combined_dual_axis(all_results):
    """Dual-axis bar charts incorporating unified scales, error bars, and despined vertical axes."""
    num_ds = len(all_results)
    fig, axes = plt.subplots(1, num_ds, figsize=(7 * num_ds, 6))
    if num_ds == 1: axes = [axes]

    fig.suptitle("Algorithmic Efficiency Profile", fontsize=16, fontweight='bold')

    # Calculate Global Maximums (including standard deviation) for uniform scaling
    global_max_time = 0
    global_max_mem = 0
    for ds_data in all_results.values():
        for m_data in ds_data.values():
            max_t = m_data["time_mean"] + m_data["time_std"]
            max_m = m_data["memory_mean"] + m_data["memory_std"]
            if max_t > global_max_time: global_max_time = max_t
            if max_m > global_max_mem: global_max_mem = max_m

    for i, (ax1, (ds_name, models_data)) in enumerate(zip(axes, all_results.items())):
        models = list(models_data.keys())
        times = [data["time_mean"] for data in models_data.values()]
        mems = [data["memory_mean"] for data in models_data.values()]

        time_stds = [data["time_std"] for data in models_data.values()]
        mem_stds = [data["memory_std"] for data in models_data.values()]

        x_indices = np.arange(len(models))
        bar_width = 0.35

        # Primary Axis (Time - Dark)
        time_colors = get_model_colors(models, "dark")
        bars_time = ax1.bar(x_indices - bar_width / 2, times, width=bar_width, color=time_colors,
                            edgecolor='black', linewidth=1, label='Time')
        ax1.errorbar(x_indices - bar_width / 2, times, yerr=time_stds, fmt='none', c='black', capsize=3, elinewidth=1.2)

        # Global Time Scale
        ax1.set_ylim(0, global_max_time * 1.10)

        # Format ax1 (Left Axis)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_color('black')
        ax1.tick_params(axis='x', colors='black')
        ax1.yaxis.grid(True, color='lightgray', linestyle='-')
        ax1.xaxis.grid(False)
        ax1.set_axisbelow(True)

        # Only show Time labels and tick marks on the far-left graph
        if i == 0:
            ax1.set_ylabel("Execution Time (s)")
            ax1.tick_params(axis='y', colors='black', left=True, labelleft=True)
        else:
            ax1.set_ylabel("")
            ax1.tick_params(axis='y', left=False, labelleft=False)

        # Secondary Axis (Memory - Light)
        ax2 = ax1.twinx()
        mem_colors = get_model_colors(models, "light")
        bars_mem = ax2.bar(x_indices + bar_width / 2, mems, width=bar_width, color=mem_colors,
                           edgecolor='black', linewidth=1, label='Memory')
        ax2.errorbar(x_indices + bar_width / 2, mems, yerr=mem_stds, fmt='none', c='black', capsize=3, elinewidth=1.2)

        # Global Memory Scale
        ax2.set_ylim(0, global_max_mem * 1.10)

        # Format ax2 (Right Axis)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.grid(False)

        # Handle outer/inner labeling for the secondary (right) axis
        if i == num_ds - 1:
            ax2.set_ylabel("Peak Memory (MB)")
            ax2.tick_params(axis='y', colors='black', right=False)
        else:
            ax2.set_ylabel("")
            ax2.tick_params(axis='y', labelright=False, right=False)

        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(models, fontweight='bold')
        ax1.set_title(ds_name, fontsize=14)

        if i == 0:
            time_proxy = mpatches.Patch(color='#777777', label='Execution Time (Dark Tones)')
            mem_proxy = mpatches.Patch(color='#b3b3b3', label='Peak Memory (Light Tones)')
            ax1.legend(handles=[time_proxy, mem_proxy], loc='upper left', frameon=True)

        if num_ds > 1:
            axes[1].tick_params(axis='y', left=False, right=False, labelleft=False)

    plt.tight_layout()
    plt.savefig(f"{figure_directory}dual_axis_metrics_combined.png", dpi=600)
    plt.show()


def plot_2d_embeddings(all_results):
    """Grid of 2D scatter plots for manifold inspection."""
    for ds_name, models_data in all_results.items():
        num_models = len(models_data)
        if num_models == 0:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"2D Projections: {ds_name}", fontsize=16, fontweight='bold')
        axes_flat = axes.flatten()

        i = -1
        for i, (model_name, data) in enumerate(models_data.items()):
            if i >= len(axes_flat):
                break

            ax = axes_flat[i]
            X_2d = data["X_train_reduced"]
            labels = data["y_train"]
            is_monochromatic = len(np.unique(labels)) <= 1

            scatter = ax.scatter(
                X_2d[:, 0], X_2d[:, 1],
                c=labels, cmap="tab10" if not is_monochromatic else None, alpha=0.6, s=2
            )
            ax.set_title(model_name, fontsize=12)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if not is_monochromatic and ds_name == "Fashion-MNIST":
                legend = ax.legend(*scatter.legend_elements(), title="Classes", loc="upper right", fontsize='small')
                ax.add_artist(legend)

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_xticks([])
            axes_flat[j].set_yticks([])
            axes_flat[j].set_title("(Empty Slot)", fontsize=10, color="gray")
            axes_flat[j].spines['top'].set_visible(False)
            axes_flat[j].spines['right'].set_visible(False)
            axes_flat[j].spines['left'].set_visible(False)
            axes_flat[j].spines['bottom'].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{figure_directory}{ds_name.lower().replace(' ', '_')}_2d_embeddings.png", dpi=600)
        plt.show()


def plot_classification_metrics(all_results, baselines):
    """Line charts comparing downstream Accuracy and F1 across dimensions with baseline markers."""
    num_ds = len(all_results)
    fig, axes = plt.subplots(2, num_ds, figsize=(6 * num_ds, 9), sharex=True)
    if num_ds == 1: axes = np.array([axes]).T

    fig.suptitle("Downstream Classification Signal Preservation", fontsize=16, fontweight='bold')

    # Feature size lookup based on the project proposal specifications
    dims_map = {
        "Fashion-MNIST": "784 dims",
        "PBMC 3k": "~32,000 dims"
    }

    for col_idx, (ds_name, dim_data) in enumerate(all_results.items()):
        # Dynamically harvest whatever dimensions exist in the results (e.g., 2, 3, 4, 5)
        sorted_dims = sorted([int(d) for d in dim_data.keys()])
        models = list(dim_data[sorted_dims[0]].keys())
        feat_str = dims_map.get(ds_name, "Full dims")

        # --- Row 0: Accuracy ---
        ax_acc = axes[0, col_idx]
        b_acc = baselines[ds_name]["acc_mean"]

        # Thinner horizontal dotted line for baseline
        ax_acc.axhline(y=b_acc, color='gray', linestyle=':', linewidth=1.2)
        # Contextual label next to the dotted line near the right margin
        ax_acc.text(max(sorted_dims) * 0.85, b_acc + (b_acc * 0.01), f"Baseline ({feat_str})",
                    color='gray', fontsize=9, fontweight='semibold')

        # --- Row 1: Macro-F1 ---
        ax_f1 = axes[1, col_idx]
        b_f1 = baselines[ds_name]["f1_mean"]
        ax_f1.axhline(y=b_f1, color='gray', linestyle=':', linewidth=1.2)
        ax_f1.text(max(sorted_dims) * 0.85, b_f1 + (b_f1 * 0.01), f"Baseline ({feat_str})",
                   color='gray', fontsize=9, fontweight='semibold')

        for model_name in models:
            if model_name == "t-SNE": continue
            color = COLOR_MAP.get(model_name, {"dark": "#333333"})["dark"]

            acc_means = [dim_data[d][model_name]["acc_mean"] for d in sorted_dims]
            acc_stds = [dim_data[d][model_name]["acc_std"] for d in sorted_dims]
            f1_means = [dim_data[d][model_name]["f1_mean"] for d in sorted_dims]
            f1_stds = [dim_data[d][model_name]["f1_std"] for d in sorted_dims]

            # Line plots matching original aesthetic style
            ax_acc.errorbar(sorted_dims, acc_means, yerr=acc_stds, fmt='-o', color=color,
                            linewidth=2, elinewidth=1.2, capsize=3, label=model_name)
            ax_f1.errorbar(sorted_dims, f1_means, yerr=f1_stds, fmt='-o', color=color,
                           linewidth=2, elinewidth=1.2, capsize=3, label=model_name)

        # Titles and Formatting
        ax_acc.set_title(f"{ds_name} (Accuracy)", fontsize=13)
        ax_f1.set_title(f"{ds_name} (Macro-F1)", fontsize=13)

        ax_f1.set_xlabel("Target Dimensions", fontweight='bold')
        ax_f1.set_xticks(sorted_dims)
        ax_f1.set_xticklabels([str(d) for d in sorted_dims])

        if col_idx == 0:
            ax_acc.set_ylabel("Accuracy Score")
            ax_f1.set_ylabel("Macro-F1 Score")
            ax_acc.legend(loc="lower right", frameon=True)

        format_standard_axes(ax_acc)
        format_standard_axes(ax_f1)

    plt.tight_layout()
    plt.savefig(f"{figure_directory}classification_signal_linecharts.png", dpi=600)
    plt.show()


def plot_unsupervised_metrics(all_results):
    """Line charts comparing cluster separation (Silhouette) and neighborhood preservation (Trustworthiness)."""
    num_ds = len(all_results)
    fig, axes = plt.subplots(2, num_ds, figsize=(6 * num_ds, 9), sharex=True)
    if num_ds == 1: axes = np.array([axes]).T

    fig.suptitle("Unsupervised Manifold Structural Integrity", fontsize=16, fontweight='bold')

    for col_idx, (ds_name, dim_data) in enumerate(all_results.items()):
        sorted_dims = sorted([int(d) for d in dim_data.keys()])
        models = list(dim_data[sorted_dims[0]].keys())

        ax_sil = axes[0, col_idx]
        ax_trust = axes[1, col_idx]

        for model_name in models:
            if model_name == "t-SNE": continue
            color = COLOR_MAP.get(model_name, {"dark": "#333333"})["dark"]

            sil_means = [dim_data[d][model_name]["sil_mean"] for d in sorted_dims]
            sil_stds = [dim_data[d][model_name]["sil_std"] for d in sorted_dims]
            trust_means = [dim_data[d][model_name]["trust_mean"] for d in sorted_dims]
            trust_stds = [dim_data[d][model_name]["trust_std"] for d in sorted_dims]

            ax_sil.errorbar(sorted_dims, sil_means, yerr=sil_stds, fmt='-s', color=color,
                            linewidth=2, elinewidth=1.2, capsize=3, label=model_name)
            ax_trust.errorbar(sorted_dims, trust_means, yerr=trust_stds, fmt='-s', color=color,
                              linewidth=2, elinewidth=1.2, capsize=3, label=model_name)

        ax_sil.set_title(f"{ds_name} (Silhouette Score)", fontsize=13)
        ax_trust.set_title(f"{ds_name} (Trustworthiness)", fontsize=13)

        ax_trust.set_xlabel("Target Dimensions", fontweight='bold')
        ax_trust.set_xticks(sorted_dims)
        ax_trust.set_xticklabels([str(d) for d in sorted_dims])

        if col_idx == 0:
            ax_sil.set_ylabel("Silhouette Width")
            ax_trust.set_ylabel("Trustworthiness Score")
            ax_sil.legend(loc="upper left", frameon=True)

        format_standard_axes(ax_sil)
        format_standard_axes(ax_trust)

    plt.tight_layout()
    plt.savefig(f"{figure_directory}unsupervised_metrics_linecharts.png", dpi=600)
    plt.show()
