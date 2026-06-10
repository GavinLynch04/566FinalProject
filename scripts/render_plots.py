import json
import numpy as np
from src.plotting import (
    plot_classification_metrics,
    plot_unsupervised_metrics,
    plot_time_metrics,
    plot_memory_metrics,
    plot_combined_dual_axis,
    plot_2d_embeddings
)

print("Reading experimental runs from local JSON storage backup...")
with open("metrics_results.json", "r") as f:
    raw_data = json.load(f)

baselines = raw_data["baselines"]
raw_results = raw_data["results"]

results = {}
for ds_name, dim_dict in raw_results.items():
    results[ds_name] = {}
    for dim_str, models_dict in dim_dict.items():
        dim_int = int(dim_str)
        results[ds_name][dim_int] = {}
        for model_name, metrics in models_dict.items():
            processed_metrics = {}
            for metric_key, val in metrics.items():
                # Re-densify lists back into functional numpy structures for slicing
                if metric_key in ["X_train_reduced", "X_test_reduced", "y_train", "y_test"] and val is not None:
                    processed_metrics[metric_key] = np.array(val)
                else:
                    processed_metrics[metric_key] = val
            results[ds_name][dim_int][model_name] = processed_metrics

print("[LOAD SUCCESS] In-memory structures fully restored. Generating final plots...")

plot_classification_metrics(results, baselines)
plot_unsupervised_metrics(results)

target_dim = 2
dim_2_slice = {ds_name: dataset_dims[target_dim] for ds_name, dataset_dims in results.items()}

plot_time_metrics(dim_2_slice)
plot_memory_metrics(dim_2_slice)
plot_combined_dual_axis(dim_2_slice)
plot_2d_embeddings(dim_2_slice)

print("\n[COMPLETE] All 7 production charts have been successfully rendered and updated in your /figures folder!")