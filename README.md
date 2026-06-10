# CSC566 Project: Dimensionality Reduction Comparison

This project evaluates and compares 5 different feature reduction techniques across high-dimensional datasets: PCA, t-SNE, Isomap, UMAP, and a Variational Autoencoder (VAE). It measures computational constraints (execution time, peak memory usage) alongside structural conservation metrics (Silhouette Score, Trustworthiness) and downstream logistic regression classification accuracy across target dimensions 2, 3, 4, and 5.

---

## Prerequisites and Installation

Ensure Python installed along with the required libraries. They can be installed using pip:

```bash
pip install numpy scipy scikit-learn matplotlib seaborn torch torchvision umap-learn scanpy

```

---

## Codebase Repository Directory

### Root Directory

* **`main.py`**: Executes full pipeline, formatting console tables and saving `metrics_results.json`.
* **`test_run.py`**: Runs diagnostic checks utilizing `helper.py` mock development data.

### `scripts/`

* **`extract_summary.py`**: Parses `metrics_results.json` to output evaluation metrics.
* **`render_plots.py`**: Re-reads `metrics_results.json` to trigger functions defined in `plotting.py`.

### `src/`

* **`Autoencoder.py`**: Custom PyTorch Variational Autoencoder with scikit-learn API.
* **`DataLoader.py`**: Loads and caches Fashion-MNIST and PBMC 3k datasets.
* **`evaluation.py`**: Calculates classification accuracy, Macro-F1, silhouette, and trustworthiness scores.
* **`helper.py`**: Stores experiment constants and mock data utilities for simulated development.
* **`plotting.py`**: Contains layout configurations for all 7 experimental performance charts.
* **`profiling.py`**: Measures model execution time and peak memory footprint.

---

## How to Configure and Run Experiments

### 1. Launching the Main Experiment (with Custom Runs)

Run the primary pipeline script. You can optionally specify the number of cross-validation iterations using the `--num_runs` flag (defaults to 10):

```bash
python main.py --num_runs 5
```

### 2. Executing a Quick Test

Run the diagnostic script with mock development data to verify that environment dependencies and plotting paths function correctly without waiting for full calculations:

```bash
python test_run.py
```

### 3. Extracting Scores and Regenerating Visuals

Analyze results or update graphics post-experiment without retraining the models:

* **Re-render all saved figures:**
  ```bash
  python scripts/render_plots.py
  ```
* **View a clean breakdown of numerical metrics:**
  ```bash
  python scripts/extract_summary.py
  ```



---

## Project Outputs

Once an execution successfully finishes, the directory will update with the following artifacts:

* **`metrics_results.json`**: A structured JSON database containing all validation scores, standard deviations, execution constraints, and raw coordinate matrices for every single evaluated model configuration.
* **`figures/` Directory**: A directory containing all plots:
  * `classification_signal_linecharts.png`: Line charts tracking accuracy and F1 trends against full feature baselines.
  * `unsupervised_metrics_linecharts.png`: Neighborhood preservation and clustering cohesion lines.
  * `dual_axis_metrics_combined.png`: Shared resource profiles charting time vs. memory overhead.
  * `[dataset_name]_2d_embeddings.png`: Grouped scatter plots displaying spatial layout distributions.