import numpy as np

# --- Constants ---
DIMENSIONS        = [2, 3, 4, 5]
MODEL_NAMES       = ["PCA", "t-SNE", "Isomap", "UMAP", "VAE"]
N_RUNS            = 10
SUBSAMPLE_SIZE    = 7500
TEST_SIZE         = 0.2


def subsample_dataset(X, y, max_size):
    """Downsamples a dataset if it exceeds max_size."""
    if max_size and X.shape[0] > max_size:
        print(f"  Subsampling dataset from {X.shape[0]} to {max_size}.")
        idx = np.random.choice(X.shape[0], max_size, replace=False)
        return X[idx], y[idx]
    return X, y


def generate_mock_results(ds_name, model_name, num_samples, n_components):
    """
    Generates realistic mock metrics for a single (dataset, model, dim, run) combination.
    Returns the same keys that profiling.py would produce so fast_dev follows
    the full downstream path.
    """
    baselines = {
        "Fashion-MNIST": {
            "PCA":    (0.05,  9.46),
            "Isomap": (0.98,  104.24),
            "UMAP":   (33.43, 79.65),
        },
        "PBMC 3k": {
            "PCA":    (0.37,  140.17),
            "Isomap": (2.07,  80.32),
            "UMAP":   (19.33, 191.43),
        },
    }

    mean_time, mean_mem = baselines.get(ds_name, {}).get(model_name, (1.0, 50.0))
    n_classes   = 10 if ds_name == "Fashion-MNIST" else 8
    train_size  = int(num_samples * (1 - TEST_SIZE))
    test_size   = num_samples - train_size

    return {
        "time":        max(0.001, np.random.normal(mean_time, mean_time * 0.1)),
        "memory_mb":   max(1.0,   np.random.normal(mean_mem,  mean_mem  * 0.1)),
        "X_train_reduced": np.random.normal(size=(train_size, n_components)).astype(np.float32),
        "X_test_reduced":  np.random.normal(size=(test_size,  n_components)).astype(np.float32),
        "y_train":     np.random.randint(0, n_classes, train_size),
        "y_test":      np.random.randint(0, n_classes, test_size),
    }