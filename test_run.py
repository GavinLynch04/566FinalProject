from main import run_pipeline, save_results_to_json
from src.evaluation import print_results_table
from src.plotting import (
    plot_classification_metrics,
    plot_unsupervised_metrics,
    plot_time_metrics,
    plot_memory_metrics,
    plot_combined_dual_axis,
    plot_2d_embeddings
)

# Run the pipeline (returns the full nested results across all dimensions: 2, 3, 4, 5)
results, baselines = run_pipeline(num_runs=1, fast_dev=True)

# Gather results
print_results_table(results, baselines)
save_results_to_json(results, baselines)

# Plot Results
plot_classification_metrics(results, baselines)
plot_unsupervised_metrics(results)

target_dim = 2
dim_2_slice = {ds_name: ds_data[target_dim] for ds_name, ds_data in results.items()}

plot_time_metrics(dim_2_slice)
plot_memory_metrics(dim_2_slice)
plot_combined_dual_axis(dim_2_slice)
plot_2d_embeddings(dim_2_slice)