import time

import umap
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from DataLoader import *


MODELS = [PCA, TSNE, Isomap, umap.UMAP]


if __name__ == "__main__":
    DL = ProjectDataLoader()
    X, y = DL.load_fashion_mnist()

    results = []
    for model in MODELS:
        '''Isomap and tSNE do not have options to transform new data, making 
        test train splits impossible before fitting. they are unsupervised, so it 
        could be okay, but we need to decide if we split after or choose different methods.'''

        model_name = type(model).__name__
        print(f"Running {model_name}...")

        start_time = time.perf_counter()

        X_reduced = model.fit_transform(X)

        end_time = time.perf_counter()

        results[model_name] = {
            "time": end_time - start_time,
            "reduced_data": X_reduced
        }

        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, y, test_size=0.2, random_state=42)

