# IDH prediction
This is the official code base for paper `Real Time Intradialytic Hypotension Forecasting Using Multimodal Deep Learning`


## Data Availability

Because the dataset includes sensitive information, access will be provided only upon request and after a formal Data Access Agreement (DAA) has been signed. This process ensures adherence to institutional review board (IRB) requirements, participant consent provisions, and relevant privacy laws such as HIPAA and GDPR. The data contain potentially identifiable and/or clinical details that could carry a risk of re-identification, even after de-identification procedures. Therefore, a controlled access framework enables the authors to safeguard participant confidentiality, oversee appropriate data use, and confirm that any secondary analyses remain consistent with the original consent and study objectives. Researchers requesting access must demonstrate a valid scientific rationale, comply with usage restrictions, and implement appropriate data security measures. This approach aims to balance research transparency and reproducibility with the responsibility to protect participant privacy and institutional standards.

For the detailed permission to access the data, please contact [jst@sanderlingllc.com](mailto:jst@sanderlingllc.com) and [r3malhotra@health.ucsd.edu](mailto:r3malhotra@health.ucsd.edu).

## Code Availability

The analytical code and executable version of the model for direct inference can be made available upon request within the context of research collaboration.

The code is made available for educational and research use; for permission to use it in such contexts, please contact [trahman@ucsd.edu](mailto:trahman@ucsd.edu).

## Data structure

Data, trained weights, and saved results are **not distributed with this repository**. The pipeline is:
**raw tables → patient records → samples + session history + care records → patient-level folds → training/results**.
Use consistently pseudonymized integer patient/session IDs across all inputs; filenames must not contain extra underscores.

### 1. Raw tables

Place your own exports under `raw_data/` (or set `IDH_RAW_ROOT`). CSV files have headers. Dates use `%m/%d/%Y %H:%M`, for example `01/01/2000 08:30`. These are neutral filenames for the original CSV/tab-separated export format.

| File | Required fields / format |
| --- | --- |
| `demographics.csv` | `PATIENT_ID`, `DATE_OF_BIRTH`, `RACE`, `SEX` |
| `lab_results.csv` | `PATIENT_ID`, `LAB_DATE_TIME`, `LAB_TEST_RESULT`, `PROCEDURE_CODE` |
| `lab_codes.tsv` | Tab-separated, **no header**: test name in column 0, procedure code in column 1; code types must match `PROCEDURE_CODE` |
| `session_details/*.csv` | One row per measurement: `PATIENT_ID`, `SESSION_ID`, `SESSION_DETAIL_ID`, `DATETIME`, `SYSBP`, `DIASBP`, `PULSE`, `PATIENT_TEMP`, `PATIENT_TEMP_UNITS`, `BLOOD_FLOW`, `DIALYSATE_FLOW`, `DIALYSATE_TEMP`, `UFR`, `NET_VOLUME_REMOVED`, `COMMENTS`. Reader uses `cp1252` encoding. |
| `session_summaries/*.csv` | One row per completed session: `PATIENT_ID`, `DIAL_STOP_TIME`, and the 18 numeric fields in `src/data_prepare/prepare_summaries.py` (`sum_keys[:18]`, in that order). |
| `diagnoses.csv` | `PATIENT_ID`, `ENTRY_DATE_TIME`, `DIAG_DESCRIPTION` |

Measurements are numeric, with missing numeric values represented as NaN. Cleaning converts patient temperatures marked `C` to Fahrenheit. Preserve the original measurement units; normalization divisors are in `DENOS` and `denos` in the preparation scripts.

Also supply `data/saline_symptoms.csv`: columns `Note`, `Saline`, `Symptoms`, `Supine`, `Volume_back`. `Note` matches `COMMENTS` exactly; annotation columns contain numeric `0`/`1` values. This annotation table is an input, not automatically generated. Unmatched notes are treated as negative by the original label logic.

### 2. Intermediate and model-ready files

All files below are Python pickles, generally **without extensions**. `<patient>` and `<session>` are your pseudonymous IDs; `<index>` is an integer sample index.

| Under `data/` | Contents |
| --- | --- |
| `cleaned_records_all/<patient>` | Dictionary: `dob` (`datetime`), `race`, `sex`, `lab_test_records`, `med_records`, and `sessions` mapping session IDs to ordered lists of measurement rows. Lab entries contain `timestamp`, `result`, `test_type`. EHR preparation adds `disease` entries with `timestamp`, `disease`. The retained ingestion pipeline leaves medications empty; the core EHR pipeline uses labs and diagnoses. |
| `clean_samples_all/<patient>_<session>_<index>` | `body`: four normalized sequences; `machine`: five sequences including net volume removed; six Boolean targets (`fall20`, `fall30`, `nadir90`, `nadir100`, `hemo`, `kdoqi`); `next_measure`: four values; `acute_interv`: intervention metadata. Regular sequences are NumPy `float16` arrays inside lists. The appended zero-input session sample uses lists and session-level labels. |
| `clean_summaries_all/<patient>_<session>` | `summaries`: numeric list `[64, 18]`; `decays`: list `[64]`. Earlier sessions only, left-padded with zeros; decay is `0.997 ** elapsed_days`. |
| `cleaned_ehr_all/<patient>_<session>` | `ehrs`: numeric list `[64, 192]`; `ehr_decays`: list `[64]`, stored at float16 precision. Earlier lab/diagnosis text is encoded using ClinicalBERT and PCA; the intermediate lookup/index file is `IDH_EHR.h5`. |
| `splits_5fold_all` | List of five dictionaries, each with `train_fnames` and `test_fnames` lists. Patients are disjoint between train and test within each fold. |

The dataset loader produces float32 tensors: `bodies [8, L]` (four physiological + first four machine channels), `summaries [64, 18]`, `decays [64]`, `ehrs [64, 192]`, `ehr_decays [64]`, and `next_measures [4]`; classification labels are int64 scalars. Batching adds a leading batch dimension. `PreRealAll` uses the most recent 32 history/EHR rows internally.

`IDH_SAMPLE_EVERY` sets the resampling interval in minutes and corresponding `L`: **1→256, 3→128, 5→64, 10→32, 15→24**. Default: 1 minute. Use the same setting for preparation, training, and checkpoint loading; use separate data roots for different intervals.

## Setup

Run commands and launch notebooks **from this folder**:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies are unpinned; this is not a validated reconstruction of the historical environment. CUDA is the original model default. Set `IDH_DEVICE=cpu` for CPU use. Paths default to this folder's `raw_data/`, `data/`, and `figures/`; override them with `IDH_RAW_ROOT`, `IDH_DATA_ROOT`, and `IDH_FIGURE_ROOT` before starting Python. Configuration is in `src/config.py`.

## Prepare data

With the tables and note annotations above available, run in order:

```bash
python -m src.data_prepare.prepare_records
python -m src.data_prepare.prepare_summaries
python -m src.data_prepare.prepare_ehr
python -m src.data_prepare.prepare_data_multi all
python -m src.data_prepare.eval_split all --seed 42
```

`prepare_records` cleans measurements and writes patient records. `prepare_summaries` aligns prior numeric summaries. `prepare_ehr` loads the external `medicalai/ClinicalBERT` model and fits separate 192-component PCAs for unique lab and diagnosis sentences; each group needs at least 192 sentences. It also adds diagnoses to the local patient records. No encoder weights are bundled. `prepare_data_multi` imputes/resamples histories and creates next-measurement labels. `eval_split` creates five patient-level folds and refuses to overwrite an existing split; at least five patients are needed.

If you already have compatible model-ready files, skip preparation. Keep existing study splits when reproducing a run; newly generated folds do not recover the original unpublished split. Preserve the data version, split file, sampling interval, seeds, dependency versions, and code revision alongside each experiment.

## Models and training

`src/models/pre_real_all.py` is the multimodal multitask model; `layers.py` contains shared layers. The other non-VAE model files retain the physiological baseline and historical architecture variants. Standalone VAE experiments are omitted. The existing `VAE_Latent` helper remains because several retained models actively call it; removing it would change training and checkpoint compatibility.

```bash
python -m src.main_train_eval_multi pre_real_all 0
```

Repeat with folds `1`–`4`. The first argument names the output experiment; the entry point currently constructs `PreRealAll` regardless of this name. Defaults remain five epochs, batch size 256, learning rate `5e-4`, and weight decay `1e-5`; edit this script for an architecture/ablation change.

Outputs go to `data/exp_res/<experiment>/`: `<experiment>_model_<fold>.pt`, `<experiment>_record_<fold>` (pickle with task-keyed `y_preds` and `y_trues`), and a loss plot. Use a new experiment name to avoid overwriting results.

## Analysis and plots

| Module | Use |
| --- | --- |
| `python -m src.res_summary pre_real_all` | Summarize prediction records across all five folds, including combined definitions. |
| `python -m src.err_analysis` | Generate sex/age-group error records from checkpoints, samples, and `data/demographics_groups.pkl`: integer patient ID → `{"SEX": "M"/"F", "AGE_GROUP": 0/1/2}`. Supply your study's grouping map; it is not generated here. |
| `python -m src.inference_plot_new` | Plot a session selected from a held-out split using its fold checkpoint. Writes to `figures/pre_real_all/`. Call `plot_i(m_name=..., select_fn=..., chosen_def=...)` for a particular experiment/session. |
| `calibration.ipynb` | Fit the existing exponential calibration on a seeded 20% subset of each held-out fold and plot/evaluate on the remaining 80%. Set experiment names to your saved records. |
| `plots_results_for_1st_revision.ipynb` | Retained comparison/statistical-test, time-dependent performance, and modality plots. Set the historical experiment names to your corresponding runs; several comparisons require multiple experiments. |
| `SHAP_test.ipynb` | Generate GradientExplainer attributions from a checkpoint, save them locally, then plot feature/modality/time contributions. Optional final cells extract embeddings and run t-SNE. Set the target definition and experiment first. |

All notebook outputs and metadata have been cleared. Analysis requires your own data/results/weights. Some revision-plot cells contain historical aggregate values; replace them with your own experiment summaries when reusing those plots. The inference plot retains historical calibration constants; recalibrate them for new models.

## 📝 Citation

If you find our work helpful for your research, please consider citing the associated paper:

```
@article{TBD,
  author = {Yunfei Luo, Siwei Zhao, Subhasis Dasgupta, Joseph M Mahaffy, Peter Kotanko, Jerome Tannenbaum, Tauhidur Rahman*, Rakesh Malhotra*},
  title = {Real Time Intradialytic Hypotension Forecasting Using Multimodal Deep Learning},
  year = {2026},
  publisher = {TBD},
  url = {TBD},
  doi = {TBD},
  journal = {TBD},
}
```

## 📃 License
This project is licensed under the [Apache License 2.0](LICENSE).
