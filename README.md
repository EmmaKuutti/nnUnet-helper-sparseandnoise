Dataset creation and evaluation helper for nnU-Net.

`datasetit.py` — dataset generation
===============================

Usage for `datasetit.py`
- Run with Python launcher on Windows or other relevant environment:

```powershell
py -3 "datasetit.py" --source "C:\path\to\source_dataset" \
    --target-base "C:\path\to\target_base" --config "configuration.json"
```

Quick flags for `datasetit.py` (one-off runs):
- `--make-sparse` : generate one sparse dataset
- `--slice-step N` : keep every Nth slice in regular mode (default 4)
- `--case-step M` : process every Mth case (default 1 (all))
- `--axis {axial,coronal,sagittal}` : primary axis to sparsify (default axial)
- `--secondary-axis {axial,coronal,sagittal}` : optional second axis for regular combined sparsification
- `--secondary-slice-step N` : slice step for the secondary axis in regular mode (default 1)
- `--sparse-mode {regular,random_single_axis,random_mixed_axes}` : choose the sparsification strategy
- `--coverage-percent P` : target percentage of slices to keep in random modes (default 100)
- `--random-seed N` : optional seed for reproducible random selection
- `--make-noisy` : generate one noisy dataset
- `--noise-percent P` : percent of cases to add noise to (default from config)

Configuration (`configuration.json`) features for `datasetit.py`
- `baseline_source`: path to source folder with `imagesTr`, `labelsTr`, `dataset.json`.
- `raw_data_base`: base folder where `nnUNet_raw` will be created.
- `baseline_dataset_id` / `baseline_dataset_name`: ID and name for baseline dataset.
- `sparse_sets`: array of objects defining multiple sparse datasets. Each entry must include `dataset_id` and may include `dataset_name`, `slice_step`, `case_step`, `axis`, `secondary_axis`, `secondary_slice_step`, `ignore_label`, `sparse_mode`, `coverage_percent`, and `random_seed`. In `regular` mode, every Nth slice is selected along the chosen axes. In `random_single_axis` mode, slices are randomly chosen along the primary axis until the requested percentage is reached. In `random_mixed_axes` mode, slices are randomly chosen across axial, coronal, and sagittal orientations until the requested percentage is reached.
- `noisy_sets`: array of objects defining multiple noisy datasets. Each entry must include `dataset_id` and may include `dataset_name`, `noise_percent`, `alpha`, `sigma`, and `ignore_label`.

Example `configuration.json` for `datasetit.py`
```json
{
  "baseline_source": "C:/path/to/source_dataset",
  "raw_data_base": "C:/path/to/target_base",
  "baseline_dataset_id": 1,
  "baseline_dataset_name": "Baseline",
  "sparse_sets": [
    {
      "dataset_id": 2,
      "dataset_name": "n4_m1_axial",
      "slice_step": 4,
      "case_step": 1,
      "axis": "axial"
    },
    {
      "dataset_id": 5,
      "dataset_name": "ax8_with_sag",
      "slice_step": 8,
      "case_step": 1,
      "axis": "axial",
      "secondary_axis": "sagittal",
      "secondary_slice_step": 1
    },
    {
      "dataset_id": 6,
      "dataset_name": "random_50_percent_axial",
      "slice_step": 4,
      "case_step": 1,
      "axis": "axial",
      "sparse_mode": "random_single_axis",
      "coverage_percent": 50,
      "random_seed": 7
    },
    {
      "dataset_id": 7,
      "dataset_name": "random_40_percent_mixed",
      "slice_step": 4,
      "case_step": 1,
      "axis": "axial",
      "sparse_mode": "random_mixed_axes",
      "coverage_percent": 40,
      "random_seed": 11
    }
  ],
  "noisy_sets": [
    {
      "dataset_id": 4,
      "dataset_name": "50percentnoise",
      "noise_percent": 50,
      "alpha": 5,
      "sigma": 5
    }
  ]
}
```

Output from `datasetit.py`
- Creates `nnUNet_raw/DatasetXXX_name` folders under `raw_data_base`.
- Sparse sets are named `Dataset{ID}_n{n}_m{m}_{axis}` by default.
- Noisy sets are named `Dataset{ID}_{percent}percentnoise` by default.

`dataset_sparse_percentage.py` — random percentage-based slice sparsification
===============================

Usage for `dataset_sparse_percentage.py`
- Run with Python launcher on Windows or other relevant environment:

```powershell
py -3 "dataset_sparse_percentage.py" --source "C:\path\to\source_dataset" \
    --target-base "C:\path\to\target_base" --config "configuration_percentage.json"
```

Quick flags for `dataset_sparse_percentage.py`
- `--keep-percent P` : percentage of slices to keep (default: 50)
- `--axis {axial,coronal,sagittal}` : axis along which slices are randomly retained
- `--dataset-id N` : ID for the generated dataset
- `--dataset-name NAME` : name for the generated dataset
- `--ignore-label N` : label value used for removed voxels
- `--random-seed N` : optional seed for reproducible randomness

Configuration (`configuration_percentage.json`) fields
- `baseline_source`: source dataset folder containing `imagesTr`, `labelsTr`, and `dataset.json`
- `raw_data_base`: base folder where `nnUNet_raw` will be created
- `dataset_id`: output dataset ID
- `dataset_name`: output dataset name
- `keep_percent`: percentage of slices to retain
- `axis`: axis to sparsify along
- `ignore_label`: label value to use for removed voxels
- `random_seed`: optional seed for reproducible random selection

`results.py` — evaluation and metrics
===============================

Usage for `results.py`
- Run with Python launcher on Windows or other relevant environment using a configuration file:

```powershell
py -3 "results.py" --conf "C:\path\to\configuration.json"
```

- Or use command-line flags directly (useful for one-off runs):

```powershell
py -3 "results.py" --pred-base "C:\path\to\predictions" --gt-dir "C:\path\to\labels" --raw-base "C:\path\to\raw" --dataset 1:Baseline --dataset 2:Sparse
```

- Or mix both (flags override configuration file):

```powershell
py -3 "results.py" --conf "C:\path\to\configuration.json" --pred-base "C:\custom\predictions"
```

Quick flags for `results.py`:
- `--conf <path>` or `--config <path>` : path to configuration.json (optional when using other flags)
- `--pred-base <path>` or `--nnunet-testresults-base <path>` : base directory for nnU-Net prediction results
- `--gt-dir <path>` or `--testset-labels-dir <path>` : directory containing ground truth test labels
- `--raw-base <path>` or `--raw-data-base <path>` : base directory where nnUNet_raw is located
- `--dataset id:name` : dataset to evaluate in format 'id:name' (can be used multiple times for multiple datasets)

Configuration needed for `results.py`
- `nnunet_testresults_base`: base folder where predicted nnU-Net result folders live.
- `testset_labels_dir`: path to ground truth test labels for evaluation.
- `raw_data_base`: base folder used by `datasetit.py`, so class labels can be read from dataset JSON.

Example `configuration.json` additions for `results.py`
```json
{
  "nnunet_testresults_base": "C:/path/to/nnunet_testresults",
  "testset_labels_dir": "C:/path/to/test_set/labelsTs",
  "raw_data_base": "C:/path/to/target_base"
}
```

Evaluation output from `results.py`
- For each configured dataset, it computes per-class Dice, Precision, Recall, Volumetric Ratio, and HD95.
- Writes `results_Dataset{ID}_{Name}.json` in the prediction folder.
- Writes `results_Dataset{ID}_{Name}.csv` next to the JSON file.
- The JSON summary includes mean/median values and per-case metrics for all computed scores.
- After all datasets are processed, it also writes `results_all_datasets.csv` and `results_all_datasets.json` in `nnunet_testresults_base`.
- The combined files collect the mean and median Dice and HD95 values for every dataset and class in one place.
- After all datasets are processed, it also writes two cross-dataset histogram-style figures in `nnunet_testresults_base`:
  - `results_histogram_dice.png` for per-class mean and median Dice values across datasets.
  - `results_histogram_hd95.png` for per-class mean and median HD95 values across datasets.
- The histogram plots label each series with the dataset id and dataset name, and each subplot title includes the class name and class id.

Notes
- `datasetit.py` copies images unchanged and sparsifies or warps labels depending on the operation.
- `results.py` pulls class names from each dataset's `dataset.json` if available, otherwise it infers classes from ground truth labels.

Contact
- Edit `configuration.json` to customize datasets, then run the appropriate script.

---