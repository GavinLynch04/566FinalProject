import numpy as np
import time
import tracemalloc
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import umap
from Autoencoder import VariationalAutoencoder


def get_models(n_components):
    return {
        "PCA":    PCA(n_components=n_components),
        "Isomap": Isomap(n_components=n_components, n_jobs=-1),
        "UMAP": umap.UMAP(n_components=n_components, n_jobs=-1),
        "VAE": VariationalAutoencoder(n_components=n_components, epochs=25, device="cpu"),
    }


def profile_single_run(model_instance, X_train, X_test):
    tracemalloc.start()
    start = time.perf_counter()
    try:
        X_train_reduced = model_instance.fit_transform(X_train)
        X_test_reduced = model_instance.transform(X_test)
        elapsed     = time.perf_counter() - start
        _, peak_mem = tracemalloc.get_traced_memory()
        return {
            "time":        elapsed,
            "memory_mb":   peak_mem / (1024 * 1024),
            "X_train_reduced": X_train_reduced,
            "X_test_reduced":  X_test_reduced,
        }
    except Exception as e:
        print(f"    -> Run failed: {e}")
        return None
    finally:
        tracemalloc.stop()