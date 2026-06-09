# Early-Prediction-of-Deterioration-HF-Patients
ML for early prediction of deterioration for Heart Failure (HF) patients based on Home Hospitalization data.

Early Prediction of Deterioration in Heart Failure Patients

Author: Ethel Kobrin

This repository contains the code and supporting artifacts for a machine-learning project that predicts acute clinical deterioration in heart failure (HF) patients enrolled in home hospitalization service. The model uses vital signs recorded during home visits — systolic and diastolic blood pressure, pulse, oxygen saturation, and weight — and predicts, at each visit, the probability that the patient will experience an acute hospitalization or death within the following 14 days.

Headline Result

Under patient-grouped 5-fold cross-validation on 21,516 visits across 904 HF patients, the tuned XGBoost model reaches AUC 0.751 ± 0.015 (nested CV AUC 0.772 ± 0.024). When the continuous output is mapped to outcome-defined risk tiers, the High tier flags 8.9% of visits and contains a 14-day deterioration rate of 46.5% — roughly seven times the Low-tier rate. Across patients who deteriorated, 90% were placed in the High or Moderate tier at least once before their event.

Repository Contents
- heart_failure_prediction_FINAL.py    Main pipeline (≈1,600 lines, 14 numbered sections)
-  README.md                      This file
-  requirements.txt               Python dependencies

The pipeline is implemented as a single script with numbered sections corresponding to the stages of the analysis:
SectionPurpose
1. Data loading from home hospitalization exports (HTML-disguised .xls format)
2. Hospitalization event extraction and merging with manual review labels
3. Patient-level dataset construction
4. Visit labeling (imminent / gray zone / stable)
5. Vital sign extraction from Hebrew free-text notes
6. Unified dataset assembly (produces both training and with-gray-zone files)
7. Feature engineering (baselines, rolling means, deltas, drift)
8. Feature evaluation (coverage, univariate AUC, distributions, correlations)
9. Logistic Regression baseline + SHAP
10. Tuned XGBoost classifier + SHAP + gray-zone reach
11. Patient-level risk-tier construction and validation
12. Vital-sign trajectory analysis across days-to-event
13. Feature ablations (current-value, no-delta, baseline-only)
14. Nested cross-validation and calibration analysis.

Flags at the top of the script toggle the heavier validation steps on and off:
- pythonRUN_FEATURE_EVAL   = True
- RUN_BASELINE_LR    = True
- RUN_XGBOOST        = True
- RUN_RISK_TIERS     = True
- RUN_TRAJECTORY     = True
- RUN_ABLATIONS      = False   # ~3 minutes additional
- RUN_NESTED_CV      = False   # ~7 minutes additional

With ablations and nested CV off, the full pipeline runs in under a minute. With all flags enabled, the complete validation run takes approximately ten minutes on a standard laptop.

Data

The patient master file, hospitalization status records, home-visit notes, and the manual hospitalization-review file are stored in a restricted-access Google Drive folder. Access is granted on request:
Data folder: https://drive.google.com/drive/folders/1krb2l5PhK_ULdkMA2re_mDro7rCguEGp?usp=sharing

The folder structure expected by the script is:

data/
1) raw                           Source files from home hospitalization organization (anonymized)
- patients.xls
- crm-...statuses.xls
-  crm-...visits-*.xls         (11 visit files)
-   anual reviews/
       - hospitalization_review.xlsx
2) processed/                      Generated outputs (created by the pipeline)
    - patient_level_dataset.csv
    - visits_features.csv
    - visits_features_with_gray.csv
    - visits_extracted.csv
    - visits_unified.csv
Paths to the data and figure directories are set at the top of heart_failure_prediction.py and should be adjusted to match local copies.

Running the Pipeline

Requirements:
Python 3.11 or later. Install dependencies:
bashpip install -r requirements.txt
Required libraries: pandas, numpy, scikit-learn, xgboost, shap, matplotlib, openpyxl, lxml.
Execution
bashpython heart_failure_prediction.py
All outputs (figures and result CSVs) are written to the figures/ directory and to data/processed/. Random seeds are fixed throughout the pipeline; re-running on the same data produces identical results.

Method in One Paragraph

For each home visit, personalized features are constructed relative to the patient's own clinical history rather than against population thresholds: per-vital baselines (median of stable visits), 14-day and 30-day rolling means computed from strictly prior visits, deviation of the current measurement from baseline, and the drift between the 14-day and 30-day means. Together with age and sex this yields 27 features. The model is a tuned XGBoost classifier (max_depth=3, learning_rate=0.01, n_estimators=300, reg_lambda=5), trained with native missing-value handling and a class-weighted loss to address the ~9% positive base rate. Evaluation uses patient-grouped 5-fold cross-validation; the headline AUC is confirmed unbiased via nested cross-validation.
The model's continuous output is reframed as three outcome-defined risk tiers (Low / Moderate / High) at cutoffs 0.50 and 0.60 on a time-smoothed combined risk score. This formulation avoids both the alarm-fatigue problem of per-visit binary alerts and the calibration miscalibration of class-weighted probabilities.

Project Documents

The full thesis report, including methodology, results, and discussion, is provided separately as part of the project submission.
