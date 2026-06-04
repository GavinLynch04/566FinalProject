from main import run_pipeline, save_results_to_json
from evaluation import print_results_table
from plotting import (
    plot_classification_metrics,
    plot_unsupervised_metrics,
    plot_time_metrics,
    plot_memory_metrics,
    plot_combined_dual_axis,
    plot_2d_embeddings
)

# 1. Run the pipeline (returns the full nested results across all dimensions: 2, 3, 4, 5)
results, baselines = run_pipeline(num_runs=1, fast_dev=True)

# 2. Print the console table summary
print_results_table(results, baselines)

# 3. Save a hard copy of the data to JSON
save_results_to_json(results, baselines)

# 4. Plot the line charts (These expect the FULL nested dictionary to draw lines across dims)
plot_classification_metrics(results, baselines)
plot_unsupervised_metrics(results)

# 5. FIX THE NESTING BUG: Extract a flat model slice for a single dimension (Dimension 2)
# Your team's bar charts and scatter plots can only display one dimension level at a time.
target_dim = 2
dim_2_slice = {ds_name: ds_data[target_dim] for ds_name, ds_data in results.items()}

# 6. Run the remaining charts using the corrected flat data slice
plot_time_metrics(dim_2_slice)
plot_memory_metrics(dim_2_slice)
plot_combined_dual_axis(dim_2_slice)
plot_2d_embeddings(dim_2_slice)