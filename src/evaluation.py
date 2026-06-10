import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from helper import (DIMENSIONS, MODEL_NAMES, N_RUNS, SUBSAMPLE_SIZE,
                    TEST_SIZE, subsample_dataset, generate_mock_results)
from profiling import get_models, profile_single_run

from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Fits logistic regression and returns (accuracy, macro-F1)."""
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return accuracy_score(y_test, preds), f1_score(y_test, preds, average="macro")


def evaluate_dataset(ds_name, X, y, num_runs=N_RUNS, fast_dev=False):
    """
    Runs the full sweep for one dataset.
    Returns:
        baseline  : (mean_acc, std_acc, mean_f1, std_f1)
        dim_data  : { n_components -> { model_name -> aggregated_metrics_dict } }
    """
    X, y = subsample_dataset(X, y, SUBSAMPLE_SIZE)

    baseline, dim_data = _run_all(ds_name, X, y, num_runs, fast_dev)
    return baseline, dim_data


def _run_all(ds_name, X, y, num_runs, fast_dev):
    """Inner loop: num_runs iterations, each with a fresh random split."""
    run_acc = {d: {m: [] for m in MODEL_NAMES} for d in DIMENSIONS}
    run_f1 = {d: {m: [] for m in MODEL_NAMES} for d in DIMENSIONS}
    run_time = {d: {m: [] for m in MODEL_NAMES} for d in DIMENSIONS}
    run_mem = {d: {m: [] for m in MODEL_NAMES} for d in DIMENSIONS}
    run_sil = {d: {m: [] for m in MODEL_NAMES} for d in DIMENSIONS}
    run_trust = {d: {m: [] for m in MODEL_NAMES} for d in DIMENSIONS}
    base_accs, base_f1s = [], []

    # Store last run's reduced arrays for d=2 plotting
    last_reduced = {m: None for m in MODEL_NAMES}

    for run in range(num_runs):
        seed = run
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
        )

        # Baseline on full-dimensional data
        acc_b, f1_b = train_and_evaluate(X_train, y_train, X_test, y_test)
        base_accs.append(acc_b)
        base_f1s.append(f1_b)

        for d in DIMENSIONS:
            for model_name, model_instance in get_models(d).items():
                if model_instance is None:
                    continue  # Safely bypasses t-SNE for dimensions 3, 4, and 5

                mode_str = "[MOCK]" if fast_dev else "[REAL]"
                print(f"  {mode_str} Dataset: {ds_name} | Run {run + 1}/{num_runs} | Dimension: {d} | Model: {model_name}...", flush=True)

                if fast_dev:
                    mock = generate_mock_results(ds_name, model_name, X.shape[0], d)
                    result = {k: mock[k] for k in ("time", "memory_mb", "X_train_reduced", "X_test_reduced")}
                    y_tr, y_te = mock["y_train"], mock["y_test"]
                else:
                    X_tr, y_tr = X_train, y_train

                    # Densify for models that can't handle sparse input
                    if sp.issparse(X_tr):
                        X_tr = X_tr.toarray()
                    if sp.issparse(X_test):
                        X_test = X_test.toarray()

                    result = profile_single_run(model_instance, X_tr, X_test)

                    y_te = y_test

                    if result is None:
                        continue

                acc, f1 = train_and_evaluate(
                    result["X_train_reduced"], y_tr,
                    result["X_test_reduced"],  y_te
                )
                run_acc[d][model_name].append(acc)
                run_f1[d][model_name].append(f1)
                run_time[d][model_name].append(result["time"])
                run_mem[d][model_name].append(result["memory_mb"])

                # Calculate Unsupervised Metrics on test set
                try:
                    sil = silhouette_score(result["X_test_reduced"], y_te)
                    # Trustworthiness needs original high-dim test data for comparison
                    trust = trustworthiness(X_test, result["X_test_reduced"], n_neighbors=5)
                except ValueError:
                    # Failsafe in case a model outputs NaNs or compresses to a single point
                    sil, trust = 0.0, 0.0

                run_sil[d][model_name].append(sil)
                run_trust[d][model_name].append(trust)

                # Keep last run's 2D arrays for plotting
                if d == 2:
                    last_reduced[model_name] = {
                        "X_train_reduced": result["X_train_reduced"],
                        "X_test_reduced":  result["X_test_reduced"],
                        "y_train":     y_tr,
                        "y_test":      y_te,
                    }

    baseline = _aggregate_baseline(base_accs, base_f1s)
    dim_data = _aggregate_runs(run_acc, run_f1, run_time, run_mem, run_sil, run_trust, last_reduced)
    return baseline, dim_data


def _aggregate_baseline(accs, f1s):
    return {
        "acc_mean": np.mean(accs), "acc_std": np.std(accs),
        "f1_mean":  np.mean(f1s),  "f1_std":  np.std(f1s),
    }


def _aggregate_runs(run_acc, run_f1, run_time, run_mem, run_sil, run_trust, last_reduced):
    """Collapses per-run lists into mean/std dicts."""
    dim_data = {}
    for d in DIMENSIONS:
        dim_data[d] = {}
        for model_name in MODEL_NAMES:
            accs   = run_acc[d][model_name]
            f1s    = run_f1[d][model_name]
            times  = run_time[d][model_name]
            mems   = run_mem[d][model_name]
            sils   = run_sil[d][model_name]
            trusts = run_trust[d][model_name]
            if not accs:
                continue
            entry = {
                "acc_mean":    np.mean(accs),   "acc_std":    np.std(accs),
                "f1_mean":     np.mean(f1s),    "f1_std":     np.std(f1s),
                "time_mean":   np.mean(times),  "time_std":   np.std(times),
                "memory_mean": np.mean(mems),   "memory_std": np.std(mems),
                "sil_mean":    np.mean(sils),   "sil_std":    np.std(sils),
                "trust_mean":  np.mean(trusts), "trust_std":  np.std(trusts),
            }
            if d == 2 and last_reduced[model_name]:
                entry.update(last_reduced[model_name])
            dim_data[d][model_name] = entry
    return dim_data


def print_results_table(all_results, baselines):
    col_w  = [18, 6, 10, 10, 10, 10, 10, 10, 10, 10]
    header = ["Model", "Dims",
              "Time", "±",
              "Mem", "±",
              "Acc", "±",
              "F1", "±"]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    fmt = "|" + "|".join(f"{{:<{w}}}" for w in col_w) + "|"

    for ds_name, dim_data in all_results.items():
        b = baselines[ds_name]
        print(f"\n{'=' * (sum(col_w) + len(col_w) + 1)}")
        print(f"  {ds_name}  |  Baseline — "
              f"Acc: {b['acc_mean']:.4f} ±{b['acc_std']:.4f}  "
              f"F1: {b['f1_mean']:.4f} ±{b['f1_std']:.4f}")
        print(sep)
        print(fmt.format(*header))
        print(sep)

        for d, model_data in dim_data.items():
            for model_name, m in model_data.items():
                drop = m["acc_mean"] - b["acc_mean"]
                row = [
                    model_name, str(d),
                    f"{m['time_mean']:.2f}",   f"{m['time_std']:.2f}",
                    f"{m['memory_mean']:.1f}",  f"{m['memory_std']:.1f}",
                    f"{m['acc_mean']:.4f}",     f"{m['acc_std']:.4f}",
                    f"{m['f1_mean']:.4f}",      f"{m['f1_std']:.4f}",
                ]
                print(fmt.format(*row))
            print(sep)