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

        shutil.copy(json_file, target_path / "dataset.json")
        return target_path

    def make_sparse(self, source_dataset_dir=None, raw_data_base=None, slice_step=4, case_step=1, ignore_label=None, axis='axial', dataset_id=2, dataset_name=None):
        """Create `nnUNet_raw/DatasetXXX_n...` where for every m-th case
        we keep every n-th slice along the chosen axis. Non-selected cases are filled with ignore_label.

        Parameters:
        - slice_step: keep every `slice_step`-th slice (n)
        - case_step: process every `case_step`-th case (m)
        - ignore_label: label value used for ignored voxels
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
        axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
        if axis not in axis_map:
            raise ValueError(f"Unknown axis '{axis}'. Choose from {list(axis_map.keys())}.")
        if dataset_name is None:
            dataset_name = f"n{slice_step}_m{case_step}_{axis}"
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

        # Infer ignore label if not provided: max existing label + 1
        inferred_ignore = None
        if original_json.get("labels") and isinstance(original_json["labels"], dict):
            int_vals = [v for v in original_json["labels"].values() if isinstance(v, int)]
            max_label = max(int_vals) if int_vals else 0
            inferred_ignore = max_label + 1
        if ignore_label is None:
            ignore_label_value = inferred_ignore if inferred_ignore is not None else 255
        else:
            ignore_label_value = ignore_label
        for idx, label_file in enumerate(label_files):
            label_img = nib.load(str(label_file))
            # Use integer dtype that can hold inferred ignore label
            data = label_img.get_fdata().astype(np.int16)

            # Prepare sparse label volume filled with ignore label
            sparse_data = np.full(data.shape, int(ignore_label_value), dtype=np.int16)

            # Build slicing tuple to select along the chosen axis
            ax = axis_map[axis]
            sel_slices = [slice(None), slice(None), slice(None)]
            if (idx % case_step) == 0:
                sel_slices[ax] = slice(None, None, slice_step)
                sel_slices = tuple(sel_slices)
                sparse_data[sel_slices] = data[sel_slices]

            new_img = nib.Nifti1Image(sparse_data, label_img.affine, label_img.header)
            nib.save(new_img, target_path / "labelsTr" / label_file.name)

            # Copy corresponding image (assumes _0000 suffix)
            img_name = label_file.name.replace('.nii.gz', '_0000.nii.gz')
            src_img = images_dir / img_name
            if src_img.exists():
                shutil.copy(src_img, target_path / "imagesTr" / img_name)

        # Copy and adapt dataset.json: preserve original style, add ignore label
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
            # Preserve any existing region ordering or other keys
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
            if 'numTraining' in data:
                data['numTraining'] = total_cases
            if ignore_label is not None:
                if 'labels' not in data or not isinstance(data['labels'], dict):
                    data['labels'] = {'background': 0}
                data['labels']['ignore'] = int(ignore_label)
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
        "--case-step",
        type=int,
        default=1,
        help="Process every Mth case (default: 1 = all)"
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
        help="Slice step for secondary axis when using combined sparsification (default: 1)"
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
                case_step=set_cfg.get("case_step", args.case_step),
                ignore_label=set_cfg.get("ignore_label", args.ignore_label),
                axis=set_cfg.get("axis", args.axis),
                secondary_axis=set_cfg.get("secondary_axis", args.secondary_axis),
                secondary_slice_step=set_cfg.get("secondary_slice_step", args.secondary_slice_step),
                dataset_id=dataset_id,
                dataset_name=dataset_name,
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
            case_step=args.case_step,
            ignore_label=args.ignore_label,
            axis=args.axis,
            secondary_axis=args.secondary_axis,
            secondary_slice_step=args.secondary_slice_step,
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
