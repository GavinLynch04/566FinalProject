import os
import torch
import numpy as np
import scanpy as sc
from torchvision import datasets, transforms
from sklearn.datasets import make_swiss_roll, make_moons
from sklearn.preprocessing import StandardScaler


class ProjectDataLoader:
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def load_fashion_mnist(self):
        raw_path = os.path.join(self.data_dir, 'FashionMNIST')
        exists = os.path.exists(raw_path)

        if exists:
            print(f"Found Fashion-MNIST in {raw_path}. Skipping download.")
        else:
            print("Fashion-MNIST not found. Starting download...")

        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.FashionMNIST(root=self.data_dir, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root=self.data_dir, train=False, download=True, transform=transform)

        X = torch.cat([train_set.data, test_set.data], dim=0).numpy()
        y = torch.cat([train_set.targets, test_set.targets], dim=0).numpy()

        X = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
        return X, y

    def load_pbmc_3k(self):
        file_path = os.path.join(self.data_dir, 'pbmc3k_raw.h5ad')

        if os.path.exists(file_path):
            print(f"Found PBMC 3k at {file_path}. Loading local file.")
            adata = sc.read_h5ad(file_path)
        else:
            print("PBMC 3k not found. Downloading...")
            adata = sc.datasets.pbmc3k()
            adata.write(file_path)

        # scanpy preprocessing, might need to change this
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        y = np.zeros(X.shape[0])
        return X, y

    def load_synthetic(self, n_samples=3000, type='swiss_roll', noise=0.1):
        if type == 'swiss_roll':
            X, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
        else:
            X, _ = make_moons(n_samples=n_samples, noise=noise)

        X = StandardScaler().fit_transform(X)
        return X.astype(np.float32)


# for autoencoder
def get_torch_loader(X, batch_size=128, shuffle=True):
    tensor_x = torch.Tensor(X)
    dataset = torch.utils.data.TensorDataset(tensor_x)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)