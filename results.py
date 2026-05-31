import csv
import os
import json
import argparse
import numpy as np
import SimpleITK as sitk
from medpy.metric.binary import hd95, dc


def load_config(path="configuration.json"):
    with open(path, "r", encoding="utf-8") as f:
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


def read_labels_from_dataset_json(raw_data_base, dataset_id, dataset_name):
    dataset_json_path = os.path.join(raw_data_base, "nnUNet_raw", f"Dataset{dataset_id:03d}_{dataset_name}", "dataset.json")
    if os.path.exists(dataset_json_path):
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
    class_scores = {cid: {"dice": [], "hd95": []} for cid in classes_mapping.keys()}

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

            class_scores[cid]["dice"].append({"case": fname, "value": dice_score})

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
        hd95s = [c["value"] for c in class_scores[cid]["hd95"] if c["value"] is not None]

        summary[cid] = {
            "name": cname,
            "cases_evaluated": len(dices),
            "mean_dice": float(np.mean(dices)) if dices else None,
            "median_dice": float(np.median(dices)) if dices else None,
            "mean_hd95": float(np.mean(hd95s)) if hd95s else None,
            "median_hd95": float(np.median(hd95s)) if hd95s else None,
            "per_case": {
                "dice": class_scores[cid]["dice"],
                "hd95": class_scores[cid]["hd95"]
            }
        }

    return summary


def write_csv_summary(output_csv_path, dataset_id, dataset_name, summary):
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["dataset_id", "dataset_name", "class_id", "class_name", "cases_evaluated", "mean_dice", "median_dice", "mean_hd95", "median_hd95"])
        for cid, info in summary.items():
            writer.writerow([
                dataset_id,
                dataset_name,
                cid,
                info.get("name"),
                info.get("cases_evaluated"),
                info.get("mean_dice"),
                info.get("median_dice"),
                info.get("mean_hd95"),
                info.get("median_hd95"),
            ])


def main(config_path="configuration.json"):
    cfg = load_config(config_path)

    pred_base = cfg.get("nnunet_testresults_base")
    gt_dir = cfg.get("testset_labels_dir")
    raw_base = cfg.get("raw_data_base")

    if not pred_base or not gt_dir or not raw_base:
        raise ValueError("configuration.json must include 'nnunet_testresults_base', 'testset_labels_dir', and 'raw_data_base'")

    datasets = get_dataset_list(cfg)
    if not datasets:
        print("No datasets found in configuration.json")
        return

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

        out_name = os.path.join(pred_dir, f"results_Dataset{did:03d}_{dname}.json")
        with open(out_name, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        csv_name = os.path.join(pred_dir, f"results_Dataset{did:03d}_{dname}.csv")
        write_csv_summary(csv_name, did, dname, summary)

        print(f"Wrote results to {out_name}")
        print(f"Wrote CSV summary to {csv_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate nnU-Net predictions and compute metrics")
    parser.add_argument(
        "--conf",
        "--config",
        dest="config_path",
        default="configuration.json",
        help="Path to configuration.json (default: configuration.json)"
    )
    args = parser.parse_args()
    main(config_path=args.config_path)
