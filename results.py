import csv
import os
import json
import argparse
import numpy as np
import SimpleITK as sitk
from medpy.metric.binary import hd95, dc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def resolve_input_path(path):
    if not path:
        return None

    path = os.path.expanduser(str(path).strip().strip('"').strip("'"))

    # Fix common Windows typo where a path becomes "C:C:\..." instead of "C:\..."
    if os.name == "nt" and len(path) >= 4 and path[1] == ":" and path[2] == path[0] and path[3] == ":":
        path = path[2:]

    if os.path.isabs(path):
        return os.path.normpath(path)

    for base in [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]:
        candidate = os.path.normpath(os.path.join(base, path))
        if os.path.exists(candidate):
            return candidate

    return os.path.normpath(path)


def load_config(path="configuration.json"):
    resolved_path = resolve_input_path(path)
    if not resolved_path:
        raise FileNotFoundError("No configuration path provided")

    with open(resolved_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dataset_list(cfg):
    datasets = []
    # baseline
    if "baseline_dataset_id" in cfg:
        name = cfg.get("baseline_dataset_name", "Baseline")
        datasets.append({"id": cfg["baseline_dataset_id"], "name": name})

    # sparse
    for s in cfg.get("sparse_sets", []):
        datasets.append({"id": s["dataset_id"], "name": s.get("dataset_name", f"Dataset{s['dataset_id']}")})

    # noisy
    for s in cfg.get("noisy_sets", []):
        datasets.append({"id": s["dataset_id"], "name": s.get("dataset_name", f"Dataset{s['dataset_id']}")})

    return datasets


def find_dataset_json_path(raw_data_base, dataset_id, dataset_name):
    raw_data_base = os.path.normpath(str(raw_data_base))
    candidate_roots = [
        raw_data_base,
        os.path.join(raw_data_base, "nnUNet_raw"),
        os.path.join(raw_data_base, "nnUNet_raw", "nnUNet_raw"),
    ]
    candidate_roots = [os.path.normpath(root) for root in candidate_roots]

    candidate_name = f"Dataset{dataset_id:03d}_{dataset_name}"
    dataset_name_lower = str(dataset_name).lower() if dataset_name is not None else ""

    for root in candidate_roots:
        if not os.path.isdir(root):
            continue

        dataset_json_path = os.path.join(root, candidate_name, "dataset.json")
        if os.path.exists(dataset_json_path):
            return dataset_json_path

        prefix = f"Dataset{dataset_id:03d}_"
        candidates = [name for name in os.listdir(root) if name.startswith(prefix)]
        if not candidates:
            continue

        for name in candidates:
            if dataset_name_lower and dataset_name_lower in name.lower():
                candidate = os.path.join(root, name, "dataset.json")
                if os.path.exists(candidate):
                    return candidate

        for name in candidates:
            candidate = os.path.join(root, name, "dataset.json")
            if os.path.exists(candidate):
                return candidate

    return None


def read_labels_from_dataset_json(raw_data_base, dataset_id, dataset_name):
    dataset_json_path = find_dataset_json_path(raw_data_base, dataset_id, dataset_name)
    if dataset_json_path and os.path.exists(dataset_json_path):
        with open(dataset_json_path, "r", encoding="utf-8") as f:
            dj = json.load(f)
        labels = dj.get("labels", {})
        # Build mapping: try to convert keys to ints, skip non-numeric keys (e.g., "background")
        mapping = {}
        for k, v in labels.items():
            try:
                key_int = int(k)
                if key_int != 0:
                    mapping[key_int] = v
            except (ValueError, TypeError):
                # Skip non-numeric keys (e.g., "background", "ignore", etc.)
                pass
        return mapping if mapping else None
    return None


def evaluate_dataset(pred_dir, gt_dir, classes_mapping):
    # classes_mapping: {class_id: name}
    class_scores = {cid: {"dice": [], "hd95": [], "precision": [], "recall": [], "volumetric_ratio": []} for cid in classes_mapping.keys()}

    files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")])
    for fname in files:
        pred_path = os.path.join(pred_dir, fname)
        gt_path = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            print(f"Warning: missing ground truth for {fname}; skipping")
            continue

        gt_img = sitk.ReadImage(gt_path)
        pred_img = sitk.ReadImage(pred_path)

        spacing_flipped = gt_img.GetSpacing()[::-1]
        gt_arr = sitk.GetArrayFromImage(gt_img)
        pred_arr = sitk.GetArrayFromImage(pred_img)

        for cid, cname in classes_mapping.items():
            gt_bin = (gt_arr == cid)
            pred_bin = (pred_arr == cid)

            if not np.any(gt_bin):
                # class not present in GT for this case -> skip scoring for this case
                continue

            # Dice: if pred empty but gt has, dice = 0
            try:
                dice_score = float(dc(pred_bin.astype(int), gt_bin.astype(int)))
            except Exception:
                dice_score = 0.0

            tp = int(np.logical_and(pred_bin, gt_bin).sum())
            fp = int(np.logical_and(pred_bin, np.logical_not(gt_bin)).sum())
            fn = int(np.logical_and(np.logical_not(pred_bin), gt_bin).sum())
            gt_volume = int(gt_bin.sum())
            pred_volume = int(pred_bin.sum())

            precision_score = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall_score = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            volumetric_ratio = float(pred_volume / gt_volume) if gt_volume > 0 else None

            class_scores[cid]["dice"].append({"case": fname, "value": dice_score})
            class_scores[cid]["precision"].append({"case": fname, "value": precision_score})
            class_scores[cid]["recall"].append({"case": fname, "value": recall_score})
            class_scores[cid]["volumetric_ratio"].append({"case": fname, "value": volumetric_ratio})

            # HD95: if pred empty -> treat as undefined (store null)
            if not np.any(pred_bin):
                class_scores[cid]["hd95"].append({"case": fname, "value": None})
            else:
                try:
                    h = float(hd95(pred_bin, gt_bin, voxelspacing=spacing_flipped))
                    class_scores[cid]["hd95"].append({"case": fname, "value": h})
                except Exception:
                    class_scores[cid]["hd95"].append({"case": fname, "value": None})

    # summarize
    summary = {}
    for cid, cname in classes_mapping.items():
        dices = [c["value"] for c in class_scores[cid]["dice"]]
        precisions = [c["value"] for c in class_scores[cid]["precision"]]
        recalls = [c["value"] for c in class_scores[cid]["recall"]]
        volumetric_ratios = [c["value"] for c in class_scores[cid]["volumetric_ratio"] if c["value"] is not None]
        hd95s = [c["value"] for c in class_scores[cid]["hd95"] if c["value"] is not None]

        summary[cid] = {
            "name": cname,
            "cases_evaluated": len(dices),
            "mean_dice": float(np.mean(dices)) if dices else None,
            "median_dice": float(np.median(dices)) if dices else None,
            "mean_precision": float(np.mean(precisions)) if precisions else None,
            "median_precision": float(np.median(precisions)) if precisions else None,
            "mean_recall": float(np.mean(recalls)) if recalls else None,
            "median_recall": float(np.median(recalls)) if recalls else None,
            "mean_volumetric_ratio": float(np.mean(volumetric_ratios)) if volumetric_ratios else None,
            "median_volumetric_ratio": float(np.median(volumetric_ratios)) if volumetric_ratios else None,
            "mean_hd95": float(np.mean(hd95s)) if hd95s else None,
            "median_hd95": float(np.median(hd95s)) if hd95s else None,
            "per_case": {
                "dice": class_scores[cid]["dice"],
                "precision": class_scores[cid]["precision"],
                "recall": class_scores[cid]["recall"],
                "volumetric_ratio": class_scores[cid]["volumetric_ratio"],
                "hd95": class_scores[cid]["hd95"]
            }
        }

    return summary


def write_csv_summary(output_csv_path, dataset_id, dataset_name, summary):
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["dataset_id", "dataset_name", "class_id", "class_name", "cases_evaluated", "mean_dice", "median_dice", "mean_precision", "median_precision", "mean_recall", "median_recall", "mean_volumetric_ratio", "median_volumetric_ratio", "mean_hd95", "median_hd95"])
        for cid, info in summary.items():
            writer.writerow([
                dataset_id,
                dataset_name,
                cid,
                info.get("name"),
                info.get("cases_evaluated"),
                info.get("mean_dice"),
                info.get("median_dice"),
                info.get("mean_precision"),
                info.get("median_precision"),
                info.get("mean_recall"),
                info.get("median_recall"),
                info.get("mean_volumetric_ratio"),
                info.get("median_volumetric_ratio"),
                info.get("mean_hd95"),
                info.get("median_hd95"),
            ])


def write_combined_summary(output_csv_path, output_json_path, dataset_results):
    rows = []
    for dataset in dataset_results:
        dataset_id = dataset["dataset_id"]
        dataset_name = dataset["dataset_name"]
        for cid, info in dataset["summary"].items():
            rows.append({
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "class_id": cid,
                "class_name": info.get("name"),
                "cases_evaluated": info.get("cases_evaluated"),
                "mean_dice": info.get("mean_dice"),
                "median_dice": info.get("median_dice"),
                "mean_hd95": info.get("mean_hd95"),
                "median_hd95": info.get("median_hd95"),
            })

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["dataset_id", "dataset_name", "class_id", "class_name", "cases_evaluated", "mean_dice", "median_dice", "mean_hd95", "median_hd95"])
        for row in rows:
            writer.writerow([
                row["dataset_id"],
                row["dataset_name"],
                row["class_id"],
                row["class_name"],
                row["cases_evaluated"],
                row["mean_dice"],
                row["median_dice"],
                row["mean_hd95"],
                row["median_hd95"],
            ])

    with open(output_json_path, "w", encoding="utf-8") as jsonfile:
        json.dump({"rows": rows}, jsonfile, indent=2)


def format_dataset_label(dataset_id, dataset_name):
    return f"{dataset_id}: {dataset_name}"


def collect_plot_values(dataset_results, metric_name):
    values_by_class = {}
    class_names = {}
    dataset_labels = []

    for item in dataset_results:
        dataset_id = item["dataset_id"]
        dataset_name = item["dataset_name"]
        dataset_label = format_dataset_label(dataset_id, dataset_name)
        dataset_labels.append(dataset_label)

        summary = item["summary"]
        for cid, info in summary.items():
            class_names[cid] = info.get("name", f"Class_{cid}")
            metric_values = values_by_class.setdefault(cid, {"mean": {}, "median": {}})

            mean_value = info.get(f"mean_{metric_name}")
            median_value = info.get(f"median_{metric_name}")

            if mean_value is not None:
                metric_values["mean"][dataset_label] = mean_value
            if median_value is not None:
                metric_values["median"][dataset_label] = median_value

    return values_by_class, class_names, dataset_labels


def save_histogram_plot(values_by_class, class_names, dataset_labels, output_path, title, y_label):
    class_ids = [cid for cid in sorted(values_by_class.keys()) if values_by_class[cid]["mean"] or values_by_class[cid]["median"]]
    if not class_ids:
        print(f"No data available for {title}; skipping plot")
        return

    subplot_count = len(class_ids)
    ncols = 2 if subplot_count > 1 else 1
    nrows = int(np.ceil(subplot_count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(12, len(dataset_labels) * 1.6), max(5, nrows * 4)), squeeze=False)
    axes = axes.flatten()

    for idx, cid in enumerate(class_ids):
        ax = axes[idx]
        class_name = class_names.get(cid, f"Class_{cid}")
        class_values = values_by_class[cid]

        mean_values = [class_values["mean"].get(label, np.nan) for label in dataset_labels]
        median_values = [class_values["median"].get(label, np.nan) for label in dataset_labels]

        x_positions = np.arange(len(dataset_labels))
        bar_width = 0.38

        ax.bar(x_positions - bar_width / 2, mean_values, width=bar_width, color="#4C78A8", alpha=0.8, label="Mean")
        ax.bar(x_positions + bar_width / 2, median_values, width=bar_width, color="#F58518", alpha=0.8, label="Median")

        ax.set_title(f"{class_name} ({cid})")
        ax.set_ylabel(y_label)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(dataset_labels, rotation=30, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_ylim(bottom=0)
        if idx == 0:
            ax.legend(loc="best")

    for idx in range(subplot_count, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot to {output_path}")


def write_summary_plots(output_base_dir, dataset_results):
    dice_values, class_names, dataset_labels = collect_plot_values(dataset_results, "dice")
    hd95_values, _, _ = collect_plot_values(dataset_results, "hd95")

    save_histogram_plot(
        dice_values,
        class_names,
        dataset_labels,
        os.path.join(output_base_dir, "results_histogram_dice.png"),
        "Per-class Dice across datasets",
        "Dice",
    )
    save_histogram_plot(
        hd95_values,
        class_names,
        dataset_labels,
        os.path.join(output_base_dir, "results_histogram_hd95.png"),
        "Per-class HD95 across datasets",
        "HD95",
    )


def main(config_path=None, pred_base=None, gt_dir=None, raw_base=None, datasets_list=None):
    # If individual flags are provided, use them; otherwise load from config file
    if pred_base or gt_dir or raw_base or datasets_list:
        # Use flags; only load config for defaults
        if config_path and os.path.exists(config_path):
            cfg = load_config(config_path)
        else:
            cfg = {}
        
        pred_base = pred_base or cfg.get("nnunet_testresults_base")
        gt_dir = gt_dir or cfg.get("testset_labels_dir")
        raw_base = raw_base or cfg.get("raw_data_base")
        
        # Build datasets list from flags if provided, otherwise from config
        if datasets_list:
            datasets = []
            for ds_str in datasets_list:
                parts = ds_str.split(":")
                if len(parts) == 2:
                    try:
                        ds_id = int(parts[0])
                        ds_name = parts[1]
                        datasets.append({"id": ds_id, "name": ds_name})
                    except ValueError:
                        print(f"Warning: Invalid dataset format '{ds_str}', expected 'id:name'")
        else:
            datasets = get_dataset_list(cfg)
    else:
        # Load everything from config file
        config_path = config_path or "configuration.json"
        cfg = load_config(config_path)
        pred_base = cfg.get("nnunet_testresults_base")
        gt_dir = cfg.get("testset_labels_dir")
        raw_base = cfg.get("raw_data_base")
        datasets = get_dataset_list(cfg)

    if not pred_base or not gt_dir or not raw_base:
        raise ValueError("Must provide nnunet_testresults_base, testset_labels_dir, and raw_data_base via flags or configuration.json")

    if not datasets:
        print("No datasets specified")
        return

    dataset_results = []

    for ds in datasets:
        did = ds["id"]
        dname = ds["name"]
        pred_dir = os.path.join(pred_base, f"Dataset{did:03d}_{dname}")
        if not os.path.isdir(pred_dir):
            print(f"Prediction directory missing: {pred_dir}; skipping dataset {did}")
            continue

        # try to read labels mapping from dataset.json in raw data
        classes = read_labels_from_dataset_json(raw_base, did, dname)
        if classes is None or len(classes) == 0:
            # fallback: infer classes from ground truth files (union of labels excluding 0)
            print(f"No dataset.json found for Dataset{did:03d}_{dname}, inferring classes from ground truth files...")
            classes = {}
            for f in sorted(os.listdir(gt_dir)):
                if not f.endswith('.nii.gz'):
                    continue
                arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_dir, f)))
                unique = np.unique(arr)
                for u in unique:
                    if int(u) == 0:
                        continue
                    classes.setdefault(int(u), f"Class_{int(u)}")

        print(f"Evaluating Dataset{did:03d}_{dname} with classes: {classes}")
        summary = evaluate_dataset(pred_dir, gt_dir, classes)

        out = {
            "dataset_id": did,
            "dataset_name": dname,
            "summary": summary
        }
        dataset_results.append(out)

        out_name = os.path.join(pred_dir, f"results_Dataset{did:03d}_{dname}.json")
        with open(out_name, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        csv_name = os.path.join(pred_dir, f"results_Dataset{did:03d}_{dname}.csv")
        write_csv_summary(csv_name, did, dname, summary)

        print(f"Wrote results to {out_name}")
        print(f"Wrote CSV summary to {csv_name}")

    write_summary_plots(pred_base, dataset_results)

    combined_csv_path = os.path.join(pred_base, "results_all_datasets.csv")
    combined_json_path = os.path.join(pred_base, "results_all_datasets.json")
    write_combined_summary(combined_csv_path, combined_json_path, dataset_results)
    print(f"Wrote combined CSV summary to {combined_csv_path}")
    print(f"Wrote combined JSON summary to {combined_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate nnU-Net predictions and compute metrics",
        epilog="""
Usage examples:
  # Using configuration file:
  python results.py --conf configuration.json
  
  # Using command-line flags:
  python results.py --pred-base /path/to/predictions --gt-dir /path/to/labels --raw-base /path/to/raw --dataset 1:Baseline --dataset 2:Sparse
  
  # Mix config file with flag overrides:
  python results.py --conf configuration.json --pred-base /custom/predictions
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--conf", "--config",
        dest="config_path",
        default=None,
        help="Path to configuration.json (optional if using --pred-base, --gt-dir, --raw-base flags)"
    )
    parser.add_argument(
        "--pred-base", "--nnunet-testresults-base",
        dest="pred_base",
        default=None,
        help="Base directory for nnU-Net prediction results"
    )
    parser.add_argument(
        "--gt-dir", "--testset-labels-dir",
        dest="gt_dir",
        default=None,
        help="Directory containing ground truth test labels"
    )
    parser.add_argument(
        "--raw-base", "--raw-data-base",
        dest="raw_base",
        default=None,
        help="Base directory where nnUNet_raw is located"
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=None,
        help="Dataset to evaluate in format 'id:name' (can be used multiple times, e.g., --dataset 1:Baseline --dataset 2:Sparse)"
    )
    args = parser.parse_args()
    main(
        config_path=args.config_path,
        pred_base=args.pred_base,
        gt_dir=args.gt_dir,
        raw_base=args.raw_base,
        datasets_list=args.datasets
    )
