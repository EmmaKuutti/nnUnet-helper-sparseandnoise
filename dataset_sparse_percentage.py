import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from datasetit import DatasetMaker


class PercentageSparseDatasetMaker(DatasetMaker):
    """Create a sparse dataset by randomly keeping a requested percentage of slices."""

    def _get_dataset_jobs(self):
        if not isinstance(self.config, dict):
            return []

        for key in ("sparse_sets", "datasets"):
            datasets_cfg = self.config.get(key)
            if isinstance(datasets_cfg, list) and datasets_cfg:
                jobs = []
                for entry in datasets_cfg:
                    if not isinstance(entry, dict):
                        continue
                    jobs.append({
                        "dataset_id": entry.get("dataset_id"),
                        "dataset_name": entry.get("dataset_name"),
                        "keep_percent": entry.get("keep_percent", entry.get("coverage_percent")),
                        "axis": entry.get("axis", "axial"),
                        "ignore_label": entry.get("ignore_label"),
                        "random_seed": entry.get("random_seed"),
                    })
                return jobs

        return [{
            "dataset_id": self.config.get("dataset_id", 2),
            "dataset_name": self.config.get("dataset_name"),
            "keep_percent": self.config.get("keep_percent", 50.0),
            "axis": self.config.get("axis", "axial"),
            "ignore_label": self.config.get("ignore_label"),
            "random_seed": self.config.get("random_seed"),
        }]

    def __init__(self, config_path=None):
        super().__init__(None)
        self.config_path = None
        self.config_dir = Path(__file__).resolve().parent

        script_dir = Path(__file__).resolve().parent
        cwd_dir = Path.cwd()

        if config_path is None:
            default_candidates = [
                script_dir / "configuration_percentage.json",
                cwd_dir / "configuration_percentage.json",
            ]
            for candidate in default_candidates:
                if candidate.exists():
                    config_path = candidate
                    break

        if config_path is not None:
            config_path = Path(str(config_path)).expanduser()
            candidate_paths = []
            if config_path.is_absolute():
                candidate_paths = [config_path]
            else:
                candidate_paths = [
                    (script_dir / config_path).resolve(),
                    (cwd_dir / config_path).resolve(),
                ]

            resolved_config = None
            for candidate in candidate_paths:
                if candidate.exists():
                    resolved_config = candidate
                    break
            if resolved_config is None:
                resolved_config = candidate_paths[0]
            if not resolved_config.exists():
                raise FileNotFoundError(f"Config file not found: {resolved_config}")

            self.config_path = resolved_config
            self.config_dir = self.config_path.parent
            with open(self.config_path, "r") as f:
                self.config = json.load(f)

    def _resolve_path(self, path, default=None):
        if path is None:
            path = default
        if path is None:
            return None

        path_obj = Path(str(path)).expanduser()
        if not path_obj.is_absolute():
            base_dir = self.config_dir or Path.cwd()
            path_obj = (base_dir / path_obj).resolve()
        return path_obj

    def _get_selected_slices(self, shape, axis, keep_percent=100.0, rng=None):
        axis_idx = self._get_axis_index(axis) if isinstance(axis, str) else axis
        n_slices = shape[axis_idx]

        if keep_percent is None:
            keep_percent = 100.0
        keep_percent = float(keep_percent)

        if keep_percent <= 0:
            return []
        if keep_percent >= 100.0:
            return list(range(n_slices))

        target_count = int(np.ceil(n_slices * keep_percent / 100.0))
        if target_count >= n_slices:
            return list(range(n_slices))

        if rng is None:
            rng = np.random.default_rng()

        selected = rng.choice(n_slices, size=target_count, replace=False)
        return sorted(int(v) for v in selected.tolist())

    def make_sparse_percentage(
        self,
        source_dataset_dir=None,
        raw_data_base=None,
        keep_percent=50.0,
        axis="axial",
        dataset_id=2,
        dataset_name=None,
        ignore_label=None,
        random_seed=None,
    ):
        """Create a dataset where a random subset of slices is kept.

        The selection is based on a requested percentage of slices (and therefore
        approximately the same percentage of voxels, since each slice contributes
        the same number of voxels).
        """
        source_dataset_dir = self._resolve_path(source_dataset_dir or self.config.get("baseline_source"))
        raw_data_base = self._resolve_path(raw_data_base or self.config.get("raw_data_base", "."))

        if not source_dataset_dir or not source_dataset_dir.exists():
            raise FileNotFoundError(
                f"Source dataset folder not found: {source_dataset_dir}. "
                f"Check baseline_source in {self.config_path or 'the supplied config file'}"
            )
        if not source_dataset_dir.is_dir():
            raise NotADirectoryError(f"Source dataset path is not a directory: {source_dataset_dir}")

        labels_dir = source_dataset_dir / "labelsTr"
        images_dir = source_dataset_dir / "imagesTr"
        if not labels_dir.exists():
            raise FileNotFoundError(f"Missing labelsTr in source dataset: {source_dataset_dir}")
        if not images_dir.exists():
            raise FileNotFoundError(f"Missing imagesTr in source dataset: {source_dataset_dir}")

        axis = str(axis).lower()
        self._get_axis_index(axis)

        if keep_percent is None:
            keep_percent = 100.0
        keep_percent = float(keep_percent)
        if keep_percent < 0.0 or keep_percent > 100.0:
            raise ValueError("keep_percent must be between 0 and 100")

        if dataset_name is None:
            dataset_name = f"p{int(round(keep_percent))}_{axis}"

        target_path = self._make_dataset_root(raw_data_base, dataset_id, dataset_name)
        (target_path / "imagesTr").mkdir(parents=True, exist_ok=True)
        (target_path / "labelsTr").mkdir(parents=True, exist_ok=True)

        label_files = sorted(labels_dir.glob("*.nii.gz"))
        json_file = source_dataset_dir / "dataset.json"
        original_json = {}
        if json_file.exists():
            with open(json_file, "r") as jf:
                try:
                    original_json = json.load(jf)
                except Exception:
                    original_json = {}

        inferred_ignore = None
        if "labels" in original_json:
            inferred_ignore = self._infer_ignore_label_from_labels(original_json["labels"])
        if ignore_label is None:
            ignore_label_value = inferred_ignore if inferred_ignore is not None else 255
        else:
            ignore_label_value = ignore_label

        rng = np.random.default_rng(random_seed)
        for label_file in label_files:
            label_img = nib.load(str(label_file))
            data = label_img.get_fdata().astype(np.int16)
            sparse_data = np.full(data.shape, int(ignore_label_value), dtype=np.int16)

            selected_slices = self._get_selected_slices(data.shape, axis, keep_percent=keep_percent, rng=rng)
            for slice_idx in selected_slices:
                sel = [slice(None), slice(None), slice(None)]
                sel[self._get_axis_index(axis)] = slice_idx
                sparse_data[tuple(sel)] = data[tuple(sel)]

            new_img = nib.Nifti1Image(sparse_data, label_img.affine, label_img.header)
            nib.save(new_img, target_path / "labelsTr" / label_file.name)

            img_name = label_file.name.replace(".nii.gz", "_0000.nii.gz")
            src_img = images_dir / img_name
            if src_img.exists():
                shutil.copy(src_img, target_path / "imagesTr" / img_name)

        if json_file.exists():
            with open(json_file, "r") as jf:
                try:
                    data = json.load(jf)
                except Exception:
                    data = {}

            if "labels" not in data or not isinstance(data["labels"], dict):
                data["labels"] = {"background": 0}
            data["labels"]["ignore"] = int(ignore_label_value)

            folder_name = f"Dataset{int(dataset_id):03d}_{dataset_name}"
            data["name"] = folder_name
            try:
                data["dataset_id"] = int(dataset_id)
            except Exception:
                data["dataset_id"] = dataset_id
            data["numTraining"] = len(label_files)

            with open(target_path / "dataset.json", "w") as outj:
                json.dump(data, outj, indent=4)

        return target_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a sparse dataset by randomly retaining a percentage of slices")
    parser.add_argument("--source", required=False, help="Path to the source dataset folder")
    parser.add_argument("--target-base", default=None, help="Target base folder for nnUNet_raw")
    parser.add_argument("--config", help="Path to the JSON config file to use")
    parser.add_argument("--keep-percent", type=float, default=None, help="Percentage of slices/voxels to keep (0-100)")
    parser.add_argument("--axis", type=str, default="axial", choices=["axial", "coronal", "sagittal"], help="Axis along which slices are randomly retained")
    parser.add_argument("--dataset-id", type=int, default=None, help="Output dataset ID")
    parser.add_argument("--dataset-name", default=None, help="Output dataset name")
    parser.add_argument("--ignore-label", type=int, default=None, help="Label value used for removed slices")
    parser.add_argument("--random-seed", type=int, default=None, help="Optional seed for reproducible random selection")

    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        config_path = "configuration_percentage.json"

    maker = PercentageSparseDatasetMaker(config_path)
    resolved_source = maker._resolve_path(args.source or maker.config.get("baseline_source"))
    resolved_target_base = maker._resolve_path(args.target_base or maker.config.get("raw_data_base", "."))

    if not resolved_source:
        raise ValueError("Source dataset must be provided via --source or baseline_source in the config")

    print(f"Using config file: {maker.config_path or 'no config file loaded'}")
    print(f"Source dataset: {resolved_source}")
    print(f"Target base: {resolved_target_base}")

    jobs = maker._get_dataset_jobs()
    if not jobs:
        raise ValueError("No dataset jobs were found in the config file")

    for job in jobs:
        dataset_id = args.dataset_id if args.dataset_id is not None else job.get("dataset_id")
        dataset_name = args.dataset_name if args.dataset_name is not None else job.get("dataset_name")
        keep_percent = args.keep_percent if args.keep_percent is not None else job.get("keep_percent", 50.0)
        axis = args.axis if args.axis is not None else job.get("axis", "axial")
        ignore_label = args.ignore_label if args.ignore_label is not None else job.get("ignore_label")
        random_seed = args.random_seed if args.random_seed is not None else job.get("random_seed")

        target = maker.make_sparse_percentage(
            resolved_source,
            resolved_target_base,
            keep_percent=keep_percent,
            axis=axis,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            ignore_label=ignore_label,
            random_seed=random_seed,
        )
        print(f"Created sparse dataset at: {target}")
