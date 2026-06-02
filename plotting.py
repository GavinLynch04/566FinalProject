import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

sns.set_theme(style="white")

figure_directory = "figures/"
os.makedirs(figure_directory, exist_ok=True)

# Color Map by model (Option 1: Balanced Pop)
COLOR_MAP = {
    "PCA": {"dark": "#3cb4dd", "light": "#93d6ec"},
    "t-SNE": {"dark": "#dd3c57", "light": "#ec93a2"},
    "Isomap": {"dark": "#7f3cdd", "light": "#b893ec"},
    "UMAP": {"dark": "#3cdd8c", "light": "#93ecbf"},
    "Model5": {"dark": "#dd9a3c", "light": "#ecc793"}
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
        times = [data["time"] for data in models_data.values()]
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
    plt.savefig(f"{figure_directory}execution_time_combined.png", dpi=300)
    plt.show()


def plot_memory_metrics(all_results):
    """Bar charts for peak memory footprint with standard deviation."""
    num_ds = len(all_results)
    fig, axes = plt.subplots(1, num_ds, figsize=(6 * num_ds, 5), sharey=True)
    if num_ds == 1: axes = [axes]

    fig.suptitle("Peak Memory Footprint", fontsize=15, fontweight='bold')

    for ax, (ds_name, models_data) in zip(axes, all_results.items()):
        models = list(models_data.keys())
        mems = [data["memory_mb"] for data in models_data.values()]
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
    plt.savefig(f"{figure_directory}memory_footprint_combined.png", dpi=300)
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
            max_t = m_data["time"] + m_data["time_std"]
            max_m = m_data["memory_mb"] + m_data["memory_std"]
            if max_t > global_max_time: global_max_time = max_t
            if max_m > global_max_mem: global_max_mem = max_m

    for i, (ax1, (ds_name, models_data)) in enumerate(zip(axes, all_results.items())):
        models = list(models_data.keys())
        times = [data["time"] for data in models_data.values()]
        mems = [data["memory_mb"] for data in models_data.values()]

        time_stds = [data["time_std"] for data in models_data.values()]
        mem_stds = [data["memory_std"] for data in models_data.values()]

        x_indices = np.arange(len(models))
        bar_width = 0.35

        # Primary Axis (Time - Dark)
        time_colors = get_model_colors(models, "dark")
        bars_time = ax1.bar(x_indices - bar_width / 2, times, width=bar_width, color=time_colors,
                            edgecolor='black', linewidth=1, label='Time')
        ax1.errorbar(x_indices - bar_width / 2, times, yerr=time_stds, fmt='none', c='black', capsize=3, elinewidth=1.2)

        # Apply Global Time Scale (+10% padding for visual headroom)
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

        # Apply Global Memory Scale (+10% padding for visual headroom)
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
    plt.savefig(f"{figure_directory}dual_axis_metrics_combined.png", dpi=300)
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
            X_2d = data["reduced_data"]
            labels = data["labels"]
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
        plt.savefig(f"{figure_directory}{ds_name.lower().replace(' ', '_')}_2d_embeddings.png", dpi=300)
        plt.show()