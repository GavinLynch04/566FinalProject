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

### Core Pipeline and Data

* **`DataLoader.py`**: Downloads, caches, and prepares the target data arrays. It handles loading and preprocessing for `Fashion-MNIST` (images) and `PBMC 3k` (single-cell RNA sequencing data), alongside synthetic generation configurations (`swiss_roll`, `moons`).
* **`Autoencoder.py`**: Implements the custom Variational Autoencoder (VAE) architecture in PyTorch. It provides an interface mapping to the standard scikit-learn API (`fit_transform` and `transform` methods) so it integrates with the evaluation loops.
* **`profiling.py`**: Instantiates the reduction models and tracks hardware resources. It isolates execution clock times using `time` and logs peak memory consumption in megabytes using `tracemalloc`.
* **`evaluation.py`**: Executes the cross-validation testing loop. It trains a downstream Logistic Regression model to assess classification accuracy and Macro-F1 scores, computes `silhouette_score` and `trustworthiness` matrix evaluations, and formats the ASCII summary tables printed to the terminal.
* **`plotting.py`**: Contains the configuration layouts for the project visuals. It formats and saves 7 specific charts, including resource efficiency bar charts, 2D cluster maps, and performance trajectory line graphs across dimensions.
* **`helper.py`**: Stores global experiment parameters (target dimensions, subsample capacities) and generates randomized matrices for quick testing.

### Automation

* **`main.py`**: The main script for the full experiment. It triggers data loading, loops evaluations across all models and datasets, prints final tables to the console, and serializes the raw arrays and metrics directly to a local backup file (`metrics_results.json`).
* **`test_run.py`**: A diagnostic script that executes a mock run of the code using small, random matrices. This is used to verify that the plotting tools and file paths work without waiting for heavy data calculations.
* **`render_plots.py`**: A utility script that re-reads a saved `metrics_results.json` file and regenerates the 7 image charts. This allows for the re-rendering of graphics without retraining any models.
* **`extract_summary.py`**: A data extraction tool that parses the `metrics_results.json` file and prints a clean layout of all scores across dimensions 2 through 5.

---

## How to Configure and Run Experiments

### 1. Adjusting the Number of Runs

To change how many times each model evaluates per dataset, modify `num_runs` in **`main.py`** as seen below:

```python
if __name__ == "__main__":
    # Change num_runs 
    results, baselines = run_pipeline(num_runs=10, fast_dev=False)

```

### 2. Executing a Quick Test

Run the test script (which uses mock data) to verify that the environment dependencies, formatting rules, and figure folders render without error:

```bash
python test_run.py

```

### 3. Launching the Main Experiment

To start the real experiment on the actual datasets, run:

```bash
python main.py

```

This will automatically process the calculations, print the performance layout, and save the data structures.

### 4. Extracting Scores and Regenerating Visuals

To grab raw numbers or figures without rerunning main, use these scripts directly without retraining the models:

* **To re-render all saved figures:**
```bash
python render_plots.py

```


* **To view a clean breakdown of all numerical metrics:**
```bash
python extract_summary.py

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