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
- `--slice-step N` : keep every Nth slice (default 4)
- `--case-step M` : process every Mth case (default 1 (all))
- `--axis {axial,coronal,sagittal}` : axis to sparsify (default axial)
- `--secondary-axis {axial,coronal,sagittal}` : optional second axis for combined sparsification
- `--secondary-slice-step N` : slice step for the secondary axis (default 1)
- `--make-noisy` : generate one noisy dataset
- `--noise-percent P` : percent of cases to add noise to (default from config)

Configuration (`configuration.json`) features for `datasetit.py`
- `baseline_source`: path to source folder with `imagesTr`, `labelsTr`, `dataset.json`.
- `raw_data_base`: base folder where `nnUNet_raw` will be created.
- `baseline_dataset_id` / `baseline_dataset_name`: ID and name for baseline dataset.
- `sparse_sets`: array of objects defining multiple sparse datasets. Each entry must include `dataset_id` and may include `dataset_name`, `slice_step`, `case_step`, `axis`, `secondary_axis`, `secondary_slice_step`, and `ignore_label`. When `secondary_axis` is provided, a second slicing pass is applied along that axis.
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

`results.py` — evaluation and metrics
===============================

Usage for `results.py`
- Run with Python launcher on Windows or other relevant environment:

```powershell
py -3 "results.py"
```

- Or with a custom configuration file:

```powershell
py -3 "results.py" --conf "C:\path\to\configuration.json"
```

Quick flags for `results.py`:
- `--conf <path>` or `--config <path>` : path to configuration.json (default: configuration.json in current directory)

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

Notes
- `datasetit.py` copies images unchanged and sparsifies or warps labels depending on the operation.
- `results.py` pulls class names from each dataset's `dataset.json` if available, otherwise it infers classes from ground truth labels.

Contact
- Edit `configuration.json` to customize datasets, then run the appropriate script.

---