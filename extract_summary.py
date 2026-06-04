import json

with open("metrics_results.json", "r") as f:
    data = json.load(f)

results = data["results"]
baselines = data["baselines"]

for ds_name, dim_data in results.items():
    print(f"\n==================== {ds_name.upper()} DATA SHEET ====================")
    b = baselines[ds_name]
    print(f"Full-Dimension Baseline Baseline: Acc = {b['acc_mean']:.4f} | F1 = {b['f1_mean']:.4f}")
    print("-" * 70)

    for d in sorted(dim_data.keys()):
        print(f"\n>>> TARGET DIMENSION LEVEL: {d}")
        print(f"{'Model':<10} | {'Acc':<7} | {'Macro-F1':<8} | {'Silhouette':<10} | {'Trust':<6}")
        print("-" * 55)
        for model, metrics in dim_data[d].items():
            # Handle t-SNE bypassing higher dimensions gracefully
            acc = f"{metrics.get('acc_mean', 0):.4f}" if model != "t-SNE" or d == "2" else "N/A"
            f1 = f"{metrics.get('f1_mean', 0):.4f}" if model != "t-SNE" or d == "2" else "N/A"
            sil = f"{metrics.get('sil_mean', 0):.4f}" if model != "t-SNE" else "N/A"
            trust = f"{metrics.get('trust_mean', 0):.4f}" if model != "t-SNE" else "N/A"

            print(f"{model:<10} | {acc:<7} | {f1:<8} | {sil:<10} | {trust:<6}")