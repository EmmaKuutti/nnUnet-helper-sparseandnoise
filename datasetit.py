# Code for making baseline and other nnU-Net dataset variants
import argparse
import json
import os
import shutil
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import gaussian_filter, map_coordinates

class DatasetMaker:
    def __init__(self, config_path=None):
        self.config = {}
        if config_path:
            with open(config_path, "r") as f:
                self.config = json.load(f)

    def _make_dataset_root(self, raw_data_base, dataset_id, dataset_name):
        raw_data_base = Path(raw_data_base or self.config.get("raw_data_base", "."))
        nnunet_root = raw_data_base / "nnUNet_raw"
        nnunet_root.mkdir(parents=True, exist_ok=True)
        folder_name = f"Dataset{int(dataset_id):03d}_{dataset_name}"
        dataset_path = nnunet_root / folder_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        return dataset_path

    def _infer_ignore_label_from_labels(self, labels):
        """Infer the next available ignore label from a labels mapping.

        This supports nested regional label definitions and returns the
        highest used integer plus one.
        """
        used_values = set()

        def collect_values(value):
            if isinstance(value, bool):
                return
            if isinstance(value, int):
                used_values.add(value)
            elif isinstance(value, dict):
                for v in value.values():
                    collect_values(v)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    collect_values(item)

        collect_values(labels)
        if not used_values:
            return 1
        return max(used_values) + 1

    def _normalize_sparse_mode(self, mode):
        if mode is None:
            return "regular"
        normalized = str(mode).strip().lower().replace(" ", "_")
        aliases = {
            "regular": "regular",
            "deterministic": "regular",
            "every_nth": "regular",
            "random": "random_single_axis",
            "random_single_axis": "random_single_axis",
            "random-single-axis": "random_single_axis",
            "random_mixed_axes": "random_mixed_axes",
            "random-mixed-axes": "random_mixed_axes",
            "mixed": "random_mixed_axes",
        }
        if normalized not in aliases:
            raise ValueError(f"Unknown sparse mode '{mode}'. Choose from regular, random_single_axis, or random_mixed_axes.")
        return aliases[normalized]

    def _get_axis_index(self, axis):
        axis = str(axis).lower()
        axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
        if axis not in axis_map:
            raise ValueError(f"Unknown axis '{axis}'. Choose from {list(axis_map.keys())}.")
        return axis_map[axis]

    def _get_selected_slices(self, shape, axis, mode="regular", slice_step=1, coverage_percent=100.0, rng=None, slice_start=1):
        axis_idx = self._get_axis_index(axis) if isinstance(axis, str) else axis
        n_slices = shape[axis_idx]
        start_idx = int(slice_start) - 1
        if start_idx < 0:
            raise ValueError("slice_start must be a positive integer")

        if mode == "regular":
            if slice_step <= 0:
                raise ValueError("slice_step must be a positive integer")
            return list(range(start_idx, n_slices, slice_step))

        if coverage_percent is None:
            coverage_percent = 100.0
        coverage_percent = float(coverage_percent)
        if coverage_percent <= 0:
            return []
        available_slices = list(range(start_idx, n_slices))
        if coverage_percent >= 100.0:
            return available_slices

        target_count = int(np.ceil(len(available_slices) * coverage_percent / 100.0))
        if target_count >= len(available_slices):
            return available_slices

        if rng is None:
            rng = np.random.default_rng()
        selected = rng.choice(available_slices, size=target_count, replace=False)
        return sorted(int(v) for v in selected.tolist())

    def _build_sparse_selection(self, shape, axis, secondary_axis=None, mode="regular", slice_step=1, secondary_slice_step=1, coverage_percent=100.0, rng=None, slice_start=1):
        mode = self._normalize_sparse_mode(mode)
        axis_names = [axis]
        if mode == "random_mixed_axes":
            axis_names = ["axial", "coronal", "sagittal"]
            if secondary_axis is not None and str(secondary_axis).lower() not in axis_names:
                axis_names.append(str(secondary_axis).lower())
        elif secondary_axis is not None and str(secondary_axis).lower() != str(axis).lower():
            axis_names.append(str(secondary_axis).lower())

        selection_map = {}
        if mode == "regular":
            for axis_name in axis_names:
                axis_idx = self._get_axis_index(axis_name)
                step = slice_step if axis_name == str(axis).lower() else secondary_slice_step
                selection_map[axis_idx] = self._get_selected_slices(shape, axis_idx, mode="regular", slice_step=step, coverage_percent=coverage_percent, rng=rng, slice_start=slice_start)
        elif mode == "random_single_axis":
            axis_idx = self._get_axis_index(axis)
            selection_map[axis_idx] = self._get_selected_slices(shape, axis_idx, mode="random_single_axis", slice_step=slice_step, coverage_percent=coverage_percent, rng=rng, slice_start=slice_start)
        else:
            if rng is None:
                rng = np.random.default_rng()

            if mode == "random_mixed_axes":
                axis_indices = [self._get_axis_index(axis_name) for axis_name in axis_names]
                if not axis_indices:
                    return selection_map
                chosen_axis_idx = int(rng.choice(axis_indices))
                n_slices = shape[chosen_axis_idx]
                if slice_step <= 0:
                    raise ValueError("slice_step must be a positive integer")
                start_idx = int(slice_start) - 1
                if start_idx < 0:
                    raise ValueError("slice_start must be a positive integer")
                selection_map[chosen_axis_idx] = list(range(start_idx, n_slices, slice_step))
                if not selection_map[chosen_axis_idx]:
                    if start_idx < n_slices:
                        selection_map[chosen_axis_idx] = [start_idx]
                return selection_map

            candidates = []
            start_idx = int(slice_start) - 1
            if start_idx < 0:
                raise ValueError("slice_start must be a positive integer")
            for axis_name in axis_names:
                axis_idx = self._get_axis_index(axis_name)
                for slice_idx in range(start_idx, shape[axis_idx]):
                    candidates.append((axis_idx, slice_idx))

            if not candidates:
                return selection_map

            if coverage_percent >= 100.0:
                for axis_idx in {a for a, _ in candidates}:
                    selection_map[axis_idx] = list(range(start_idx, shape[axis_idx]))
                return selection_map

            target_count = int(np.ceil(len(candidates) * float(coverage_percent) / 100.0))
            target_count = min(target_count, len(candidates))
            if target_count <= 0:
                return selection_map

            chosen = rng.choice(len(candidates), size=target_count, replace=False)
            grouped = {}
            for pos in chosen:
                axis_idx, slice_idx = candidates[int(pos)]
                grouped.setdefault(axis_idx, []).append(int(slice_idx))
            selection_map = {axis_idx: sorted(set(slice_indices)) for axis_idx, slice_indices in grouped.items()}

        return selection_map

    def _apply_sparse_selection(self, sparse_data, data, selection_map):
        for axis_idx, slice_indices in selection_map.items():
            for slice_idx in slice_indices:
                sel = [slice(None), slice(None), slice(None)]
                sel[axis_idx] = slice_idx
                sparse_data[tuple(sel)] = data[tuple(sel)]

    def _is_selected_case(self, case_index, case_step, case_start):
        return case_index >= case_start - 1 and (case_index - (case_start - 1)) % case_step == 0

    def make_baseline(self, source_dataset_dir=None, raw_data_base=None, dataset_id=1, dataset_name="Baseline"):
        """Create `nnUNet_raw` under `raw_data_base`, then copy the provided
        dataset into `nnUNet_raw/Dataset001_Baseline`.

        Parameters
        - source_dataset_dir: path to a folder that contains `imagesTr`,
          `labelsTr` and `dataset.json`.
        - raw_data_base: base folder where `nnUNet_raw` will be created.
        """
        source_dataset_dir = Path(source_dataset_dir or self.config.get("baseline_source"))
        raw_data_base = Path(raw_data_base or self.config.get("raw_data_base", "."))

        if not source_dataset_dir or not source_dataset_dir.exists():
            raise FileNotFoundError(f"Source dataset folder not found: {source_dataset_dir}")
        if not source_dataset_dir.is_dir():
            raise NotADirectoryError(f"Source dataset path is not a directory: {source_dataset_dir}")

        json_file = source_dataset_dir / "dataset.json"
        if not json_file.exists():
            raise FileNotFoundError(f"dataset.json not found in source dataset folder: {source_dataset_dir}")

        target_path = self._make_dataset_root(raw_data_base, dataset_id, dataset_name)

        for subdir in ["imagesTr", "labelsTr"]:
            src_dir = source_dataset_dir / subdir
            if not src_dir.exists():
                raise FileNotFoundError(f"Missing {subdir} in source dataset folder: {source_dataset_dir}")
            dst_dir = target_path / subdir
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in sorted(src_dir.glob("*.nii.gz")):
                shutil.copy(f, dst_dir / f.name)

        # Read original dataset.json, update dataset id/name and numTraining, then write
        data = {}
        with open(json_file, 'r') as jf:
            try:
                data = json.load(jf)
            except Exception:
                data = {}

        # Ensure labels mapping exists
        if "labels" not in data or not isinstance(data["labels"], dict):
            data["labels"] = {"background": 0}

        # Add or update identifying fields
        folder_name = f"Dataset{int(dataset_id):03d}_{dataset_name}"
        data["name"] = folder_name
        try:
            data["dataset_id"] = int(dataset_id)
        except Exception:
            data["dataset_id"] = dataset_id

        # Count training images in the created folder
        num_imgs = sum(1 for _ in (target_path / "imagesTr").glob("*.nii.gz"))
        data["numTraining"] = num_imgs

        with open(target_path / "dataset.json", 'w') as outj:
            json.dump(data, outj, indent=4)

        return target_path

    def make_sparse(self, source_dataset_dir=None, raw_data_base=None, slice_step=4, case_step=1, ignore_label=None, axis='axial', secondary_axis=None, secondary_slice_step=1, dataset_id=2, dataset_name=None, sparse_mode='regular', coverage_percent=100.0, random_seed=None, slice_start=1, case_start=1):
        """Create `nnUNet_raw/DatasetXXX_...` by keeping selected slices from one or more orientations.

        The default regular mode keeps every n-th slice along the chosen axis and
        optionally a secondary axis. Random modes instead select slices randomly
        until a target percentage of available slices is reached, while the
        mixed-axes variant chooses a single orientation at random and keeps slices
        according to the configured slice step.

        Parameters:
        - slice_step: keep every `slice_step`-th slice in regular mode (n)
        - slice_start: one-based first slice eligible for sparsification
        - case_step: process every `case_step`-th case (m)
        - case_start: one-based first case eligible for sparsification
        - ignore_label: label value used for ignored voxels
        - axis: primary axis ('axial', 'coronal', or 'sagittal')
        - secondary_axis: optional secondary axis for combined regular sparsification
        - secondary_slice_step: slice step for the secondary axis in regular mode
        - sparse_mode: 'regular', 'random_single_axis', or 'random_mixed_axes'
        - coverage_percent: percentage of slices to keep in random modes
        - random_seed: optional seed for reproducible random selection
        """
        source_dataset_dir = Path(source_dataset_dir or self.config.get("baseline_source"))
        raw_data_base = Path(raw_data_base or self.config.get("raw_data_base", "."))

        if not source_dataset_dir or not source_dataset_dir.exists():
            raise FileNotFoundError(f"Source dataset folder not found: {source_dataset_dir}")
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

        # Validate secondary axis if provided
        if secondary_axis is not None:
            secondary_axis = str(secondary_axis).lower()
            self._get_axis_index(secondary_axis)

        sparse_mode = self._normalize_sparse_mode(sparse_mode)
        slice_start = int(slice_start)
        case_start = int(case_start)
        if slice_start < 1:
            raise ValueError("slice_start must be a positive integer")
        if case_start < 1:
            raise ValueError("case_start must be a positive integer")
        if case_step <= 0:
            raise ValueError("case_step must be a positive integer")
        if sparse_mode != "regular" and secondary_axis is not None:
            secondary_axis = None

        if coverage_percent is None:
            coverage_percent = 100.0
        coverage_percent = float(coverage_percent)
        if coverage_percent < 0.0 or coverage_percent > 100.0:
            raise ValueError("coverage_percent must be between 0 and 100")

        if dataset_name is None:
            if sparse_mode == "regular":
                dataset_name = f"n{slice_step}_m{case_step}_{axis}"
            else:
                dataset_name = f"p{int(round(coverage_percent))}_{sparse_mode}_{axis}"
        target_path = self._make_dataset_root(raw_data_base, dataset_id, dataset_name)
        (target_path / "imagesTr").mkdir(parents=True, exist_ok=True)
        (target_path / "labelsTr").mkdir(parents=True, exist_ok=True)

        label_files = sorted(labels_dir.glob("*.nii.gz"))
        # Read original dataset.json to infer labels and label style
        json_file = source_dataset_dir / "dataset.json"
        original_json = {}
        if json_file.exists():
            with open(json_file, 'r') as jf:
                try:
                    original_json = json.load(jf)
                except Exception:
                    original_json = {}

        # Infer ignore label if not provided: use the highest used integer label + 1
        # Infer ignore label if not provided: use the highest used integer label + 1
        inferred_ignore = None
        if "labels" in original_json:
            inferred_ignore = self._infer_ignore_label_from_labels(original_json["labels"])
        if "labels" in original_json:
            inferred_ignore = self._infer_ignore_label_from_labels(original_json["labels"])
        if ignore_label is None:
            ignore_label_value = inferred_ignore if inferred_ignore is not None else 255
        else:
            ignore_label_value = ignore_label
        rng = np.random.default_rng(random_seed)
        for idx, label_file in enumerate(label_files):
            label_img = nib.load(str(label_file))
            # Use integer dtype that can hold inferred ignore label
            data = label_img.get_fdata().astype(np.int16)

            # Prepare sparse label volume filled with ignore label
            sparse_data = np.full(data.shape, int(ignore_label_value), dtype=np.int16)

            if self._is_selected_case(idx, case_step, case_start):
                selection_map = self._build_sparse_selection(
                    data.shape,
                    axis,
                    secondary_axis=secondary_axis,
                    mode=sparse_mode,
                    slice_step=slice_step,
                    secondary_slice_step=secondary_slice_step,
                    coverage_percent=coverage_percent,
                    rng=rng,
                    slice_start=slice_start,
                )
                self._apply_sparse_selection(sparse_data, data, selection_map)

            new_img = nib.Nifti1Image(sparse_data, label_img.affine, label_img.header)
            nib.save(new_img, target_path / "labelsTr" / label_file.name)

            # Copy corresponding image (assumes _0000 suffix)
            img_name = label_file.name.replace('.nii.gz', '_0000.nii.gz')
            src_img = images_dir / img_name
            if src_img.exists():
                shutil.copy(src_img, target_path / "imagesTr" / img_name)

        # Copy and adapt dataset.json: preserve original style, add ignore label, and set name/id
        if json_file.exists():
            with open(json_file, 'r') as jf:
                try:
                    data = json.load(jf)
                except Exception:
                    data = {}

            # Ensure labels mapping exists
            if "labels" not in data or not isinstance(data["labels"], dict):
                data["labels"] = {"background": 0}

            # Add ignore label entry
            data["labels"]["ignore"] = int(ignore_label_value)

            # Add/update identifying fields
            folder_name = f"Dataset{int(dataset_id):03d}_{dataset_name}"
            data["name"] = folder_name
            try:
                data["dataset_id"] = int(dataset_id)
            except Exception:
                data["dataset_id"] = dataset_id

            # numTraining should reflect total cases
            total_cases = len(label_files)
            data["numTraining"] = total_cases

            with open(target_path / "dataset.json", 'w') as outj:
                json.dump(data, outj, indent=4)

        return target_path

    def make_noisy(self, source_dataset_dir=None, raw_data_base=None, noise_percent=None, ignore_label=None, alpha=5, sigma=5, dataset_id=3, dataset_name=None):
        """Create `nnUNet_raw/DatasetXXX_percentnoise` by applying elastic warp noise to a fraction of cases."""
        source_dataset_dir = Path(source_dataset_dir or self.config.get("baseline_source"))
        raw_data_base = Path(raw_data_base or self.config.get("raw_data_base", "."))
        if noise_percent is None:
            noise_percent = self.config.get("noise_percent", 100)

        if not source_dataset_dir or not source_dataset_dir.exists():
            raise FileNotFoundError(f"Source dataset folder not found: {source_dataset_dir}")
        if not source_dataset_dir.is_dir():
            raise NotADirectoryError(f"Source dataset path is not a directory: {source_dataset_dir}")

        labels_dir = source_dataset_dir / "labelsTr"
        images_dir = source_dataset_dir / "imagesTr"
        if not labels_dir.exists():
            raise FileNotFoundError(f"Missing labelsTr in source dataset: {source_dataset_dir}")
        if not images_dir.exists():
            raise FileNotFoundError(f"Missing imagesTr in source dataset: {source_dataset_dir}")

        if dataset_name is None:
            percent_name = str(noise_percent).rstrip('0').rstrip('.') if noise_percent is not None else '100'
            percent_name = percent_name.replace('.', '_')
            dataset_name = f"{percent_name}percentnoise"
        target_path = self._make_dataset_root(raw_data_base, dataset_id, dataset_name)
        (target_path / "imagesTr").mkdir(parents=True, exist_ok=True)
        (target_path / "labelsTr").mkdir(parents=True, exist_ok=True)

        label_files = sorted(labels_dir.glob("*.nii.gz"))
        total_cases = len(label_files)
        percent_value = float(noise_percent)
        if percent_value <= 0:
            noisy_count = 0
        elif percent_value >= 100 or total_cases == 0:
            noisy_count = total_cases
        else:
            noisy_count = max(1, round(total_cases * percent_value / 100.0))

        noisy_indices = set(range(noisy_count))

        json_file = source_dataset_dir / "dataset.json"
        original_json = {}
        if json_file.exists():
            with open(json_file, 'r') as jf:
                try:
                    original_json = json.load(jf)
                except Exception:
                    original_json = {}

        for idx, label_file in enumerate(label_files):
            lab_path = label_file
            img_name = label_file.name.replace('.nii.gz', '_0000.nii.gz')
            img_path = images_dir / img_name
            if not img_path.exists():
                print(f"Skipping {label_file.name}: corresponding image {img_name} not found.")
                continue

            img_nii = nib.load(str(img_path))
            lab_nii = nib.load(str(lab_path))
            img_data = img_nii.get_fdata()
            lab_data = lab_nii.get_fdata().astype(np.uint8)

            if idx in noisy_indices:
                warped_lab_data = self.elastic_warp_3d(lab_data, alpha=alpha, sigma=sigma)
                out_label = warped_lab_data.astype(np.uint8)
            else:
                out_label = lab_data.astype(np.uint8)

            nib.save(nib.Nifti1Image(img_data, img_nii.affine, img_nii.header), target_path / "imagesTr" / img_name)
            nib.save(nib.Nifti1Image(out_label, lab_nii.affine, lab_nii.header), target_path / "labelsTr" / label_file.name)

        if json_file.exists():
            with open(json_file, 'r') as jf:
                try:
                    data = json.load(jf)
                except Exception:
                    data = {}

            if 'labels' not in data or not isinstance(data['labels'], dict):
                data['labels'] = {'background': 0}

            # Update numTraining to actual number of cases
            data['numTraining'] = total_cases

            # Add ignore label if requested
            if ignore_label is not None:
                data['labels']['ignore'] = int(ignore_label)

            # Add/update identifying fields
            folder_name = f"Dataset{int(dataset_id):03d}_{dataset_name}"
            data['name'] = folder_name
            try:
                data['dataset_id'] = int(dataset_id)
            except Exception:
                data['dataset_id'] = dataset_id

            with open(target_path / 'dataset.json', 'w') as outj:
                json.dump(data, outj, indent=4)

        return target_path

    def elastic_warp_3d(self, mask, alpha=5, sigma=5, seed=None):
        rng = np.random.default_rng(seed)
        shape = mask.shape
        dz = gaussian_filter(rng.standard_normal(shape), sigma) * alpha
        dy = gaussian_filter(rng.standard_normal(shape), sigma) * alpha
        dx = gaussian_filter(rng.standard_normal(shape), sigma) * alpha
        z, y, x = np.indices(shape)
        indices = np.reshape(z + dz, (-1, 1)), \
                  np.reshape(y + dy, (-1, 1)), \
                  np.reshape(x + dx, (-1, 1))
        warped = map_coordinates(mask, indices, order=0, mode='reflect').reshape(shape)
        return warped.astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy a nnU-Net dataset into Dataset001_Baseline")
    parser.add_argument(
        "--source",
        required=False,
        help="Path to the source dataset folder containing imagesTr, labelsTr, and dataset.json (overrides config)"
    )
    parser.add_argument(
        "--target-base",
        default=None,
        help="Target base folder for raw nnU-Net datasets (overrides config)"
    )
    parser.add_argument(
        "--config",
        help="Optional JSON config file with baseline_source and raw_data_base"
    )
    parser.add_argument(
        "--make-sparse",
        action="store_true",
        help="Also generate Dataset002_Sparse from the provided source"
    )
    parser.add_argument(
        "--slice-step",
        type=int,
        default=4,
        help="Keep every Nth slice (default: 4)"
    )
    parser.add_argument(
        "--slice-start",
        type=int,
        default=1,
        help="One-based first slice eligible for sparsification (default: 1)"
    )
    parser.add_argument(
        "--case-step",
        type=int,
        default=1,
        help="Process every Mth case (default: 1 = all)"
    )
    parser.add_argument(
        "--case-start",
        type=int,
        default=1,
        help="One-based first case eligible for sparsification (default: 1)"
    )
    parser.add_argument(
        "--axis",
        type=str,
        default="axial",
        help="Primary axis to sparsify: axial (z), coronal (y), sagittal (x), or combined like 'axial+sagittal'"
    )
    parser.add_argument(
        "--secondary-axis",
        type=str,
        default=None,
        choices=["axial", "coronal", "sagittal", None],
        help="Secondary axis for combined sparsification (optional)"
    )
    parser.add_argument(
        "--secondary-slice-step",
        type=int,
        default=1,
        help="Slice step for secondary axis when using combined regular sparsification (default: 1)"
    )
    parser.add_argument(
        "--sparse-mode",
        type=str,
        default="regular",
        choices=["regular", "random_single_axis", "random_mixed_axes"],
        help="Sparsification mode: regular stepping, random single-axis selection, or random mixed-axis selection"
    )
    parser.add_argument(
        "--coverage-percent",
        type=float,
        default=100.0,
        help="Target percentage of slices to keep in random sparsification modes (default: 100)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional seed for reproducible random slice selection"
    )
    parser.add_argument(
        "--ignore-label",
        type=int,
        default=None,
        help="Ignore label value used for sparse labels (default: inferred from dataset.json)"
    )
    parser.add_argument(
        "--make-noisy",
        action="store_true",
        help="Also generate Dataset003_Noisy from the provided source"
    )
    parser.add_argument(
        "--noise-percent",
        type=float,
        default=None,
        help="Percentage of cases to apply noise to (default: from config or 100)"
    )
    parser.add_argument(
        "--noise-alpha",
        type=float,
        default=5.0,
        help="Elastic warp alpha parameter (default: 5)"
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=5.0,
        help="Elastic warp sigma parameter (default: 5)"
    )
    args = parser.parse_args()

    maker = DatasetMaker(args.config)
    # Resolve paths: prefer CLI, then config, then sensible defaults
    resolved_source = args.source or maker.config.get("baseline_source")
    resolved_target_base = args.target_base or maker.config.get("raw_data_base", ".")

    if not resolved_source:
        raise ValueError("Source dataset must be provided either via --source or 'baseline_source' in configuration.json")

    baseline_dataset_id = maker.config.get("baseline_dataset_id", 1)
    baseline_dataset_name = maker.config.get("baseline_dataset_name", "Baseline")
    target = maker.make_baseline(
        resolved_source,
        resolved_target_base,
        dataset_id=baseline_dataset_id,
        dataset_name=baseline_dataset_name,
    )
    print(f"Copied baseline dataset to: {target}")

    if args.config:
        for set_cfg in maker.config.get("sparse_sets", []):
            dataset_id = set_cfg.get("dataset_id")
            if dataset_id is None:
                raise ValueError("Each sparse set in config must include a dataset_id.")
            dataset_name = set_cfg.get("dataset_name")
            sparse_target = maker.make_sparse(
                resolved_source,
                resolved_target_base,
                slice_step=set_cfg.get("slice_step", args.slice_step),
                slice_start=set_cfg.get("slice_start", args.slice_start),
                case_step=set_cfg.get("case_step", args.case_step),
                case_start=set_cfg.get("case_start", args.case_start),
                ignore_label=set_cfg.get("ignore_label", args.ignore_label),
                axis=set_cfg.get("axis", args.axis),
                secondary_axis=set_cfg.get("secondary_axis", args.secondary_axis),
                secondary_slice_step=set_cfg.get("secondary_slice_step", args.secondary_slice_step),
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                sparse_mode=set_cfg.get("sparse_mode", args.sparse_mode),
                coverage_percent=set_cfg.get("coverage_percent", args.coverage_percent),
                random_seed=set_cfg.get("random_seed", args.random_seed),
            )
            print(f"Created sparse dataset at: {sparse_target}")

        for set_cfg in maker.config.get("noisy_sets", []):
            dataset_id = set_cfg.get("dataset_id")
            if dataset_id is None:
                raise ValueError("Each noisy set in config must include a dataset_id.")
            dataset_name = set_cfg.get("dataset_name")
            noisy_target = maker.make_noisy(
                resolved_source,
                resolved_target_base,
                noise_percent=set_cfg.get("noise_percent", args.noise_percent),
                ignore_label=set_cfg.get("ignore_label", args.ignore_label),
                alpha=set_cfg.get("alpha", args.noise_alpha),
                sigma=set_cfg.get("sigma", args.noise_sigma),
                dataset_id=dataset_id,
                dataset_name=dataset_name,
            )
            print(f"Created noisy dataset at: {noisy_target}")

    if args.make_sparse and not args.config:
        sparse_target = maker.make_sparse(
            resolved_source,
            resolved_target_base,
            slice_step=args.slice_step,
            slice_start=args.slice_start,
            case_step=args.case_step,
            case_start=args.case_start,
            ignore_label=args.ignore_label,
            axis=args.axis,
            secondary_axis=args.secondary_axis,
            secondary_slice_step=args.secondary_slice_step,
            sparse_mode=args.sparse_mode,
            coverage_percent=args.coverage_percent,
            random_seed=args.random_seed,
        )
        print(f"Created sparse dataset at: {sparse_target}")

    if args.make_noisy and not args.config:
        noisy_target = maker.make_noisy(
            resolved_source,
            resolved_target_base,
            noise_percent=args.noise_percent,
            ignore_label=args.ignore_label,
            alpha=args.noise_alpha,
            sigma=args.noise_sigma,
        )
        print(f"Created noisy dataset at: {noisy_target}")
