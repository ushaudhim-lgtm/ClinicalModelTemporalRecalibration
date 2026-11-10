# Clinical Temporal Recalibration

Retrospective temporal recalibration of clinical prediction models using
longitudinally collected NHAMCS Emergency Department data (2011-2019).

Full specification: [knowledge/Clinical_Temporal_Recalibration_Project_Specification_7295.docx](knowledge/Clinical_Temporal_Recalibration_Project_Specification_7295.docx)

## Status
Baseline models trained and evaluated; temporal drift assessed on 2015-2019
(see `outputs/tables/07_temporal_drift_metrics.csv`). Next up: recalibration
strategies.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data
Raw NHAMCS ED public-use files (2011-2019) and codebook documentation have
been downloaded into `data/raw/` and `data/metadata/` from the CDC FTP server.
Files are fixed-width ASCII; column layouts must be derived from each year's
codebook PDF (variable positions are not stable across years).

## Structure
The project is a sequence of numbered Jupyter notebooks in `notebooks/`,
run in order (00 through 11), sharing common helper functions from
`notebooks/utils.py`. Each notebook writes its outputs to `data/processed/`,
`models/`, or `outputs/` so the next notebook can pick up from there.
See `configs/study_config.json` (built by `01_define_study.ipynb`) for the
outcome definition, feature list, and dev/eval year splits.
