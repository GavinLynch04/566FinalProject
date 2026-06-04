import numpy as np
import time
import tracemalloc
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import umap
from openTSNE import TSNE as OpenTSNE
from Autoencoder import VariationalAutoencoder


class TSNEWrapper:
    def __init__(self, n_components=2, random_state=42):
        self._params = {"n_components": n_components, "random_state": random_state}
        self._embedding = None

    def fit_transform(self, X):
        self._embedding = OpenTSNE(**self._params).fit(X)
        return np.array(self._embedding)

    def transform(self, X):
        return np.array(self._embedding.transform(X))


def get_models(n_components):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "PCA":    PCA(n_components=n_components),
        "Isomap": Isomap(n_components=n_components, n_jobs=-1),
        "UMAP":   umap.UMAP(n_components=n_components, n_jobs=-1),
        "t-SNE":  TSNEWrapper(n_components=n_components),
        "VAE":    VariationalAutoencoder(n_components=n_components, epochs=50, device=device),
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