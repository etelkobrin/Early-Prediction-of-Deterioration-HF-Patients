# =============================================================================
# Early Prediction of Deterioration in Heart Failure Patients
# Using Remote Monitoring Data from Home Hospitalization Company
#
# Student:    Ethel Kobrin
# Supervisor: Dr. Inbal Maayan
# Tel Aviv University
#
# This script builds the full preprocessed dataset from raw CRM files:
#   1. Loads patient master, statuses, and visit notes
#   2. Extracts and classifies hospitalization events
#   3. Merges manual HF/non-HF review labels
#   4. Builds patient-level labeled dataset
#   5. Labels each visit note (1 = imminent, 0 = stable, 8-14d = gray zone excluded)
#   6. Extracts vitals from Hebrew free text (BP, pulse, saturation, weight)
#   7. Outputs visits_unified.csv and patient_level_dataset.csv
# =============================================================================

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG — update these paths as needed
# =============================================================================
RAW_DIR        = r"C:\FINAL PROJECT\data\raw"
REVIEWS_DIR    = r"C:\FINAL PROJECT\data\raw\manual reviews"
PROCESSED_DIR  = r"C:\FINAL PROJECT\data\processed"
FIGURES_DIR    = r"C:\FINAL PROJECT\figures"

PATIENT_MASTER_FILE = "crm-26.04.26_10.23.51.xls"
STATUSES_FILE       = "crm-26.04.26_10.29.01.xls"
VISIT_FILES = [
    "crm-26.04.26_10.36.24.xls", "crm-26.04.26_10.38.34.xls", "crm-26.04.26_10.39.31.xls",
    "crm-26.04.26_13.56.44.xls", "crm-26.04.26_13.57.34.xls", "crm-26.04.26_13.59.24.xls",
    "crm-26.04.26_14.00.48.xls", "crm-26.04.26_14.44.35.xls", "crm-26.04.26_14.49.25.xls",
    "crm-26.04.26_14.49.48.xls", "crm-26.04.26_14.50.09.xls",
]
# Manual review of all hospitalization records:
#   FinalLabel = 1 → acute deterioration (count as event)
#   FinalLabel = 0 → planned/elective admission or external trauma (excluded)
# This expands the event definition from HF-only to any acute deterioration in
# an HF patient, matching the clinical reality that decompensation in this
# frail population rarely presents as isolated HF.
HOSP_REVIEW_FILE   = "hospitalization_review.xlsx"

# Labeling windows
IMMINENT_DAYS = 14       # label = 1 if event within these many days (primary window)
GRAY_ZONE_END = 21       # gray zone = (IMMINENT_DAYS, GRAY_ZONE_END); kept but excluded from training
RANDOM_STATE  = 42

# =============================================================================
# PIPELINE FLAGS — toggle which steps run in MAIN
# Set heavy/slow steps to False for day-to-day work; True for a full reproducible run.
# =============================================================================
RUN_FEATURE_EVAL    = True    # fast — coverage, AUC, distributions, correlation
RUN_BASELINE_LR     = True    # fast — Logistic Regression baseline + SHAP
RUN_XGBOOST         = True    # fast — main model + SHAP + gray-zone reach
RUN_RISK_TIERS      = True    # fast — patient-level risk tiers
RUN_TRAJECTORY      = True    # fast — vital trajectories approaching events
RUN_ABLATIONS       = False   # medium — delta / current-value / baseline-only ablations
RUN_NESTED_CV       = False   # slow (~5–10 min) — unbiased AUC + calibration

# =============================================================================
# 1. LOADERS
# =============================================================================

def _read_html_xls(path):
    """Povider's CRM exports are HTML disguised as .xls — read with pandas read_html."""
    return pd.read_html(path, encoding="utf-8")[0].dropna(how="all")


def load_patient_master(path):
    """Load patient master table (demographics, dates, outcomes)."""
    df = _read_html_xls(path)
    df.columns = [
        "PatID", "Purpose", "Dept", "HMO", "District", "SabarDistrict", "AdmissionDate",
        "City", "Sector", "Age_raw", "PhoneCATAI", "PlaceOfDeath", "EndReason",
        "ActiveDeceased", "EndDate", "DeathDate", "HospitalDischargeDate",
        "Diagnosis", "Gender", "DOB",
    ]
    df = df[pd.to_numeric(df["PatID"], errors="coerce").notna()].copy()
    df["PatID"] = df["PatID"].astype(int)
    for c in ["AdmissionDate", "EndDate", "DeathDate", "DOB"]:
        df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
    return df


def load_statuses(path):
    """Load monthly status records (where hospitalization events are documented)."""
    df = _read_html_xls(path)
    df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()]
    df.columns = ["PatID", "HospType", "TotalDays", "HospDetails", "CreateDate", "UpdateDate"]
    df["PatID"] = df["PatID"].astype(int)
    df["CreateDate"] = pd.to_datetime(df["CreateDate"], errors="coerce", dayfirst=True)
    return df


def load_visits(visit_files, raw_dir):
    """Load and combine all visit-note files."""
    parts = []
    for f in visit_files:
        df = _read_html_xls(os.path.join(raw_dir, f))
        df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()]
        df.columns = ["PatID", "VisitDate", "MainIssues", "Treatment", "Plan"]
        parts.append(df)
    visits = pd.concat(parts, ignore_index=True)
    visits["PatID"] = visits["PatID"].astype(int)
    visits["VisitDate"] = pd.to_datetime(visits["VisitDate"], errors="coerce", dayfirst=True)
    visits = visits.dropna(subset=["VisitDate"]).sort_values(["PatID", "VisitDate"]).reset_index(drop=True)
    return visits


# =============================================================================
# 2. HOSPITALIZATION EVENT EXTRACTION & LABELING
# =============================================================================

def extract_hosp_events(statuses):
    """Extract hospitalization events from status records with date + reason."""
    hosp = statuses[statuses["HospDetails"].astype(str).str.contains("אשפוז", na=False)].copy()

    def text_date(text):
        if not isinstance(text, str): return None
        m = re.search(r"תאריך[\s\d-]*אשפוז:\s*(\d{1,2}-\d{1,2}-\d{4})", text)
        if m:
            try: return pd.to_datetime(m.group(1), format="%d-%m-%Y")
            except: return None
        return None

    def reason(text):
        if not isinstance(text, str): return None
        text = re.sub(r"תאריך[\s\d-]*אשפוז:", "", text)
        m = re.search(r"סיבת אשפוז:\s*(.+?)(?:בית חולים|$)", text)
        return m.group(1).strip() if m else text.strip()

    hosp["TextDate"] = hosp["HospDetails"].apply(text_date)
    hosp["HospDate"] = hosp["TextDate"].fillna(hosp["CreateDate"])
    hosp["Reason"]   = hosp["HospDetails"].apply(reason)
    hosp["RowKey"]   = hosp["PatID"].astype(str) + "|" + hosp["Reason"].astype(str) + "|" + hosp["HospDetails"].astype(str)
    return hosp


def load_review_labels(reviews_dir):
    """
    Load the manual hospitalization-review labels (acute-deterioration definition).
    Returns a frame with RowKey and Label, where Label = FinalLabel from review.
    """
    path = os.path.join(reviews_dir, HOSP_REVIEW_FILE)
    review = pd.read_excel(path, sheet_name="Review")
    # Recompose the same RowKey used by extract_hosp_events
    review["RowKey"] = (
        review["PatID"].astype(str) + "|"
        + review["Reason"].astype(str) + "|"
        + review["HospDetails"].astype(str)
    )
    review = review.rename(columns={"FinalLabel": "Label"})
    # Drop any rows still blank (shouldn't be — your review is complete)
    review = review[review["Label"].notna()].copy()
    review["Label"] = review["Label"].astype(int)
    return review[["RowKey", "Label"]]


def get_confirmed_events(statuses, reviews_dir):
    """
    Return confirmed acute-deterioration hospitalization events
    (any acute admission in an HF patient — not only HF-specific).
    """
    hosp = extract_hosp_events(statuses)
    labels = load_review_labels(reviews_dir)
    hosp = hosp.merge(labels, on="RowKey", how="left")
    hf = hosp[hosp["Label"] == 1][["PatID", "HospDate", "Reason"]].dropna(subset=["HospDate"]).copy()
    return hf.sort_values(["PatID", "HospDate"]).reset_index(drop=True)


# =============================================================================
# 3. PATIENT-LEVEL DATASET
# =============================================================================

def build_patient_level(patients, hf_events):
    """Build the per-patient labeled dataset with demographics + outcomes."""
    # Aggregate HF events per patient
    by_pat = hf_events.groupby("PatID").agg(
        HFHospDates_list=("HospDate", lambda x: sorted([d.strftime("%Y-%m-%d") for d in x])),
        HFHosp_count=("HospDate", "count"),
        FirstHFHospDate=("HospDate", "min"),
        LastHFHospDate=("HospDate", "max"),
    ).reset_index()
    by_pat["HFHospDates"] = by_pat["HFHospDates_list"].apply(lambda lst: ";".join(lst))
    by_pat = by_pat.drop(columns=["HFHospDates_list"])

    df = patients.merge(by_pat, on="PatID", how="left")
    df["HFHosp_count"] = df["HFHosp_count"].fillna(0).astype(int)
    df["Died"]         = df["DeathDate"].notna().astype(int)
    df["HadHFHosp"]    = (df["HFHosp_count"] > 0).astype(int)
    df["Deteriorated"] = ((df["HadHFHosp"] == 1) | (df["Died"] == 1)).astype(int)
    df["EarliestEventDate"] = df[["FirstHFHospDate", "DeathDate"]].min(axis=1)

    # Age computed from DOB at admission
    df["Age"] = ((df["AdmissionDate"] - df["DOB"]).dt.days / 365.25).round(1)

    keep = [
        "PatID", "Age", "Gender", "DOB", "Sector", "HMO", "City", "District", "SabarDistrict",
        "Dept", "Diagnosis", "AdmissionDate", "EndDate", "EndReason", "ActiveDeceased",
        "Died", "DeathDate", "HadHFHosp", "HFHosp_count", "HFHospDates",
        "FirstHFHospDate", "LastHFHospDate", "Deteriorated", "EarliestEventDate",
    ]
    return df[keep]


# =============================================================================
# 4. VISIT-LEVEL LABELING
# =============================================================================

def label_visits(visits, hf_events, death_map, imminent=IMMINENT_DAYS, gray_end=GRAY_ZONE_END):
    """
    Label each visit based on the nearest upcoming deterioration event.

        Label = 1   (Horizon='imminent'):   event within `imminent` days (default 14)
        Label = NaN (Horizon='gray_zone'):  event in (imminent, gray_end] days — kept but
                                            excluded from training; used for reach analysis
        Label = 0   (Horizon='stable'):     no event within gray_end days
        Label = NaN (Horizon='after_event'/'after_death'): dropped downstream

    The Horizon column lets gray-zone visits flow through feature engineering with
    time-correct features, then be separated only at the modeling stage.
    """
    events_by_pat = hf_events.groupby("PatID")["HospDate"].apply(lambda x: sorted(x.tolist())).to_dict()

    def _label_row(row):
        pid, vdate = row["PatID"], row["VisitDate"]
        events = list(events_by_pat.get(pid, []))
        death = death_map.get(pid, pd.NaT)
        if pd.notna(death):
            events.append(death)
        events = sorted(events)

        if pd.notna(death) and vdate > death:
            return pd.Series({"Label": np.nan, "DaysToNextEvent": np.nan, "Horizon": "after_death"})
        if not events:
            return pd.Series({"Label": 0, "DaysToNextEvent": np.nan, "Horizon": "stable"})

        upcoming = [e for e in events if e >= vdate]
        if not upcoming:
            return pd.Series({"Label": np.nan, "DaysToNextEvent": np.nan, "Horizon": "after_event"})

        days_to = (upcoming[0] - vdate).days
        if days_to <= imminent:
            return pd.Series({"Label": 1, "DaysToNextEvent": days_to, "Horizon": "imminent"})
        if days_to <= gray_end:
            return pd.Series({"Label": np.nan, "DaysToNextEvent": days_to, "Horizon": "gray_zone"})
        return pd.Series({"Label": 0, "DaysToNextEvent": days_to, "Horizon": "stable"})

    labels = visits.apply(_label_row, axis=1)
    return pd.concat([visits, labels], axis=1)

# =============================================================================
# 5. VITAL SIGN EXTRACTION FROM HEBREW FREE TEXT
# =============================================================================

def _combine_text(row):
    parts = [str(row[c]) for c in ["MainIssues", "Treatment", "Plan"]
             if pd.notna(row[c]) and str(row[c]) != "nan"]
    return " | ".join(parts)


def extract_bp(text):
    """Extract first BP reading. Handles formats: XX/YY, XX\\YY, X על Y, with various labels."""
    if not isinstance(text, str): return np.nan, np.nan

    labeled = [
        r"(?:לחץ\s*דם|BP)\s*[:=]?\s*(\d{2,3})\s*[/\\]\s*(\d{2,3})",
        r"ל[\"'\.]ד\s*[:=]?\s*(\d{2,3})\s*[/\\]\s*(\d{2,3})",
        r"(?:לחץ\s*דם|BP|ל[\"'\.]ד|לד\s)\s*[:=]?\s*(\d{2,3})\s*על\s*(\d{2,3})",
        r"(?:ס[\"'\.]ח|סימנים\s*חיוניים|מדדים)\s*[:=]?\s*(\d{2,3})\s*[/\\]\s*(\d{2,3})",
    ]
    for p in labeled:
        m = re.search(p, text)
        if m:
            s, d = int(m.group(1)), int(m.group(2))
            if d > s: s, d = d, s  # auto-correct reversed order
            if 30 <= s <= 300 and 10 <= d <= 160:
                return s, d

    # Generic XX/YY only if BP-context word appears within 30 chars before
    for m in re.finditer(r"(\d{2,3})\s*[/\\]\s*(\d{2,3})", text):
        s, d = int(m.group(1)), int(m.group(2))
        if d > s: s, d = d, s
        if 30 <= s <= 300 and 10 <= d <= 160:
            pre = text[max(0, m.start() - 30):m.start()]
            if re.search(r"לחץ|דם|BP|ל[\"'\.]ד|ס[\"'\.]ח|חיוניים|מדדים", pre):
                return s, d
    return np.nan, np.nan


def extract_pulse(text):
    if not isinstance(text, str): return np.nan
    for p in [
        r"דופק\s*[:=]?\s*(\d{2,3})",
        r"\bHR\s*[:=]?\s*(\d{2,3})",
        r"pulse\s*[:=]?\s*(\d{2,3})",
        r"דופ[\"'\.]\s*[:=]?\s*(\d{2,3})",
        r"דפק\s*[:=]?\s*(\d{2,3})",
    ]:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            v = int(m.group(1))
            if 10 <= v <= 300:
                return v
    return np.nan


def extract_saturation(text):
    if not isinstance(text, str): return np.nan
    m = re.search(r"(?:סטורציה|סטורציות|סט[\"'\.]|SAT|SpO2|O2)\s*[:=]?\s*(\d{2,3})\s*%?",
                  text, re.IGNORECASE)
    if m:
        v = int(m.group(1))
        if 40 <= v <= 100:
            return v
    return np.nan


def extract_weight(text):
    if not isinstance(text, str): return np.nan
    m = re.search(r"(?:משקל|מש[\"'\.]|weight|WGT|שוקלת|שוקל)\s*[:=]?\s*(\d{2,3}(?:[.,]\d)?)",
                  text, re.IGNORECASE)
    if m:
        v = float(m.group(1).replace(",", "."))
        if 30 <= v <= 200:
            return v
    return np.nan


def extract_all_vitals(visits):
    """Add BP_sys, BP_dia, Pulse, Saturation, Weight, HasAnyVital columns."""
    visits = visits.copy()
    visits["FullText"] = visits.apply(_combine_text, axis=1)

    bp = visits["FullText"].apply(extract_bp)
    visits["BP_sys"]     = [b[0] for b in bp]
    visits["BP_dia"]     = [b[1] for b in bp]
    visits["Pulse"]      = visits["FullText"].apply(extract_pulse)
    visits["Saturation"] = visits["FullText"].apply(extract_saturation)
    visits["Weight"]     = visits["FullText"].apply(extract_weight)
    visits["HasAnyVital"] = visits[["BP_sys", "Pulse", "Saturation", "Weight"]].notna().any(axis=1).astype(int)
    return visits.drop(columns=["FullText"])


# =============================================================================
# 6. PIPELINE
# =============================================================================

def build_unified_dataset():
    """Run the full pipeline and produce the two main CSV outputs."""
    print("Loading raw data...")
    patients = load_patient_master(os.path.join(RAW_DIR, PATIENT_MASTER_FILE))
    statuses = load_statuses(os.path.join(RAW_DIR, STATUSES_FILE))
    visits   = load_visits(VISIT_FILES, RAW_DIR)
    print(f"  Patients: {len(patients):,}")
    print(f"  Statuses: {len(statuses):,}")
    print(f"  Visits:   {len(visits):,}")

    print("\nExtracting confirmed HF hospitalization events...")
    hf_events = get_confirmed_events(statuses, REVIEWS_DIR)
    print(f"  Confirmed deterioration events: {len(hf_events)} across {hf_events['PatID'].nunique()} patients")

    print("\nBuilding patient-level dataset...")
    patient_level = build_patient_level(patients, hf_events)
    print(f"  Deteriorated = 1: {(patient_level['Deteriorated']==1).sum()}")
    print(f"  Deteriorated = 0: {(patient_level['Deteriorated']==0).sum()}")

    print("\nLabeling visits...")
    visits["FirstVisitDate"] = visits.groupby("PatID")["VisitDate"].transform("min")
    visits["DaysSinceFirstVisit"] = (visits["VisitDate"] - visits["FirstVisitDate"]).dt.days
    dob_map = dict(zip(patients["PatID"], patients["DOB"]))
    visits["DOB"] = visits["PatID"].map(dob_map)
    visits["Age"] = ((visits["VisitDate"] - visits["DOB"]).dt.days / 365.25).round(1)

    death_map = dict(zip(patients["PatID"], patients["DeathDate"]))
    visits = label_visits(visits, hf_events, death_map)
    visits["Patient_Deteriorated"] = visits["PatID"].apply(
        lambda p: int(p in set(hf_events["PatID"].values) or pd.notna(death_map.get(p, pd.NaT)))
    )

    print("\nExtracting vitals from free text...")
    visits = extract_all_vitals(visits)

    # Merge demographics into visits
    demo = patients[["PatID", "Gender", "Sector", "HMO", "City", "District", "Dept", "Diagnosis"]]
    visits = visits.merge(demo, on="PatID", how="left")

    # Keep stable, imminent, AND gray-zone visits (drop only after-event / after-death).
    # Gray-zone visits carry Label = NaN but flow through feature engineering so their
    # features are time-correct; they are separated from training at the modeling stage.
    visits_final = visits[visits["Horizon"].isin(["stable", "imminent", "gray_zone"])].copy()

    cols = [
        "PatID", "VisitDate", "DaysSinceFirstVisit",
        "Label", "DaysToNextEvent", "Patient_Deteriorated",
        "Age", "Gender", "Sector", "HMO", "City", "District", "Dept", "Diagnosis",
        "BP_sys", "BP_dia", "Pulse", "Saturation", "Weight", "HasAnyVital",
        "MainIssues", "Treatment", "Plan", "Horizon",
    ]
    visits_final = visits_final[cols].sort_values(["PatID", "VisitDate"]).reset_index(drop=True)

    print(f"\n══ FINAL UNIFIED DATASET ══")
    print(f"  Rows:    {len(visits_final):,}")
    print(f"  Label = 1: {(visits_final['Label']==1).sum():,}")
    print(f"  Label = 0: {(visits_final['Label']==0).sum():,}")
    print(f"  HasAnyVital = 1: {(visits_final['HasAnyVital']==1).sum():,}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    patient_level.to_csv(os.path.join(PROCESSED_DIR, "patient_level_dataset.csv"), index=False)
    visits_final.to_csv(os.path.join(PROCESSED_DIR, "visits_unified.csv"), index=False)
    print(f"\n✓ Saved to {PROCESSED_DIR}")

    # ── Produce TWO feature files for downstream analysis ────────────────────
    # 1) Standard training file: stable + imminent only (gray zone dropped)
    visits_train = visits[visits["Horizon"].isin(["stable", "imminent"])].copy()
    visits_train["Label"] = visits_train["Label"].astype(int)

    # 2) With-gray file: stable + imminent + gray zone, time-correct features
    #    Used for the trajectory and gray-zone reach analyses
    visits_with_gray = visits[visits["Horizon"].isin(["stable", "imminent", "gray_zone"])].copy()

    return patient_level, visits_train, visits_with_gray


# =============================================================================
# 7. FEATURE ENGINEERING
# =============================================================================

VITALS = ["BP_sys", "BP_dia", "Pulse", "Saturation", "Weight"]
WINDOWS_DAYS = [14, 30]
VISIT_COUNT_WINDOWS = [7, 14]


def _patient_baselines(visits):
    """Per-patient median over all stable (Label = 0) visits."""
    stable = visits[visits["Label"] == 0]
    baselines = stable.groupby("PatID")[VITALS].median().add_suffix("_baseline")
    return baselines


def _rolling_per_patient(group, vital, window_days):
    """Rolling mean and count over `window_days` STRICTLY BEFORE each visit (closed='left')."""
    g = group.set_index("VisitDate")[vital]
    means = g.rolling(f"{window_days}D", closed="left").mean()
    counts = g.rolling(f"{window_days}D", closed="left").count()
    return pd.DataFrame({"mean": means.values, "count": counts.values}, index=g.index)


def engineer_features(visits):
    """
    Add per-visit baseline, rolling, drift, and visit-context features.

    For each vital (BP_sys, BP_dia, Pulse, Saturation, Weight) we compute:
        - <vital>_baseline           Patient's median over their stable visits.
        - <vital>_14d_mean / _count  Rolling mean and # observations in 14d prior window.
        - <vital>_30d_mean / _count  Same for 30d.
        - <vital>_delta_baseline     Current value - patient baseline.
        - <vital>_drift_14v30        14d_mean - 30d_mean (recent drift signal).

    Visit-context features:
        - DaysSinceLastVisit
        - VisitsLast7d, VisitsLast14d
    """
    df = visits.copy()
    df["VisitDate"] = pd.to_datetime(df["VisitDate"])
    df = df.sort_values(["PatID", "VisitDate"]).reset_index(drop=True)

    # Patient baselines
    df = df.merge(_patient_baselines(df), on="PatID", how="left")

    # Rolling features per vital and window
    for vital in VITALS:
        for w in WINDOWS_DAYS:
            rolled = df.groupby("PatID", group_keys=False).apply(
                lambda grp: _rolling_per_patient(grp, vital, w)
            )
            df[f"{vital}_{w}d_mean"]  = rolled["mean"].values
            df[f"{vital}_{w}d_count"] = rolled["count"].values

    # Derived
    for vital in VITALS:
        df[f"{vital}_delta_baseline"] = df[vital] - df[f"{vital}_baseline"]
        df[f"{vital}_drift_14v30"]    = df[f"{vital}_14d_mean"] - df[f"{vital}_30d_mean"]

    # Visit context
    df["DaysSinceLastVisit"] = df.groupby("PatID")["VisitDate"].diff().dt.days
    df["_one"] = 1
    for w in VISIT_COUNT_WINDOWS:
        rolled_counts = df.groupby("PatID", group_keys=False).apply(
            lambda grp: grp.set_index("VisitDate")["_one"].rolling(f"{w}D", closed="left").count()
        )
        df[f"VisitsLast{w}d"] = rolled_counts.values
    df = df.drop(columns=["_one"])

    return df


# =============================================================================
# 8. FEATURE EVALUATION (sanity checks + figures for thesis)
# =============================================================================

from sklearn.metrics import roc_auc_score

RAW_VITALS      = VITALS.copy()
BASELINE_FEATS  = [f"{v}_baseline"       for v in VITALS]
M14_FEATS       = [f"{v}_14d_mean"       for v in VITALS]
M30_FEATS       = [f"{v}_30d_mean"       for v in VITALS]
DELTA_FEATS     = [f"{v}_delta_baseline" for v in VITALS]
DRIFT_FEATS     = [f"{v}_drift_14v30"    for v in VITALS]
CONTEXT_FEATS   = ["DaysSinceLastVisit", "VisitsLast7d", "VisitsLast14d"]
ALL_FEATURES    = RAW_VITALS + BASELINE_FEATS + M14_FEATS + M30_FEATS + DELTA_FEATS + DRIFT_FEATS + CONTEXT_FEATS


def _plot_coverage(df, features, out_path):
    miss = pd.DataFrame({
        "Feature": features,
        "Coverage_%": [df[f].notna().mean() * 100 for f in features],
    }).sort_values("Coverage_%", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = ["tomato" if v < 25 else ("orange" if v < 50 else "steelblue")
              for v in miss["Coverage_%"]]
    ax.barh(miss["Feature"], miss["Coverage_%"], color=colors, edgecolor="white", alpha=0.85)
    ax.set_xlabel("Coverage (%)")
    ax.set_title("Feature Coverage in Training Data", fontweight="bold")
    ax.axvline(50, color="gray", linestyle="--", linewidth=1)
    for i, v in enumerate(miss["Coverage_%"]):
        ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return miss


def _compute_univariate_auc(df, features, label_col="Label"):
    rows = []
    for f in features:
        sub = df[[f, label_col]].dropna()
        if len(sub) < 200 or sub[label_col].nunique() < 2:
            rows.append({"Feature": f, "AUC": np.nan, "n": len(sub)})
            continue
        try:
            a1 = roc_auc_score(sub[label_col], sub[f])
            a2 = roc_auc_score(sub[label_col], -sub[f])
            rows.append({"Feature": f, "AUC": max(a1, a2), "n": len(sub)})
        except Exception:
            rows.append({"Feature": f, "AUC": np.nan, "n": len(sub)})
    return pd.DataFrame(rows).sort_values("AUC", ascending=False)


def _plot_auc(auc_df, out_path):
    plot_df = auc_df.dropna().sort_values("AUC")
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = ["tomato" if a < 0.55 else ("orange" if a < 0.6 else "steelblue")
              for a in plot_df["AUC"]]
    ax.barh(plot_df["Feature"], plot_df["AUC"], color=colors, edgecolor="white", alpha=0.85)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="No predictive power")
    ax.axvline(0.6, color="black", linestyle=":", linewidth=1, label="Useful threshold")
    ax.set_xlabel("Univariate AUC")
    ax.set_xlim(0.45, max(0.75, plot_df["AUC"].max() + 0.02))
    ax.set_title("Individual Predictive Power of Each Feature", fontweight="bold")
    ax.legend(loc="lower right")
    for i, v in enumerate(plot_df["AUC"]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_distributions(df, features, out_path, label_col="Label"):
    n = len(features)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    fig.suptitle("Feature Distributions: Stable vs Imminent", fontsize=14, fontweight="bold", y=1.00)
    axes_flat = list(axes.flat)
    for ax, feat in zip(axes_flat, features):
        s0 = df[df[label_col] == 0][feat].dropna()
        s1 = df[df[label_col] == 1][feat].dropna()
        if len(s0) == 0 or len(s1) == 0:
            ax.set_visible(False); continue
        p01 = min(s0.quantile(0.01), s1.quantile(0.01))
        p99 = max(s0.quantile(0.99), s1.quantile(0.99))
        ax.hist(s0.clip(p01, p99), bins=40, alpha=0.6, color="steelblue",
                density=True, label=f"Stable (n={len(s0):,})")
        ax.hist(s1.clip(p01, p99), bins=40, alpha=0.7, color="tomato",
                density=True, label=f"Imminent (n={len(s1):,})")
        ax.set_title(feat, fontweight="bold", fontsize=10)
        ax.set_xlabel("Value"); ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_correlation_heatmap(df, features, out_path, min_n=1000):
    feats = [f for f in features if df[f].notna().sum() > min_n]
    corr = df[feats].corr()
    fig, ax = plt.subplots(figsize=(15, 13))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot=False, linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return corr


def evaluate_features(visits_features, figures_dir=FIGURES_DIR):
    os.makedirs(figures_dir, exist_ok=True)
    print("\n[1/4] Feature coverage...")
    miss = _plot_coverage(visits_features, ALL_FEATURES,
                          os.path.join(figures_dir, "fig_features_coverage.png"))
    miss.to_csv(os.path.join(figures_dir, "feature_missingness_table.csv"), index=False)

    print("[2/4] Univariate predictive power...")
    auc_df = _compute_univariate_auc(visits_features, ALL_FEATURES)
    _plot_auc(auc_df, os.path.join(figures_dir, "fig_features_auc.png"))
    auc_df.to_csv(os.path.join(figures_dir, "feature_auc_table.csv"), index=False)

    print("[3/4] Distribution comparisons...")
    viz_feats = RAW_VITALS + DELTA_FEATS + ["DaysSinceLastVisit", "VisitsLast7d"]
    _plot_distributions(visits_features, viz_feats,
                        os.path.join(figures_dir, "fig_distributions_by_label.png"))

    print("[4/4] Correlation heatmap...")
    _plot_correlation_heatmap(visits_features, ALL_FEATURES,
                              os.path.join(figures_dir, "fig_correlation_heatmap.png"))

    print(f"\n✓ Figures and tables saved to {figures_dir}")
    return auc_df, miss


# =============================================================================
# 9. BASELINE MODEL — Logistic Regression with 5-Fold CV
# =============================================================================

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, roc_curve, precision_score,
                              recall_score, f1_score, confusion_matrix)

COVERAGE_THRESH = 0.40
N_FOLDS = 5

# Feature set: demographics (Age, Sex) + vitals only.
# We deliberately EXCLUDE visit-context features (VisitsLast7d/14d, DaysSinceLastVisit)
# and department, because the deployment target is patient self-reported app vitals;
# those features are artifacts of the visit-note data source and would not transfer.


def _build_feature_set(df, coverage_thresh=COVERAGE_THRESH):
    """
    Return (feature_columns, df_with_encoded_demographics).

    coverage_thresh filters out features below a minimum fraction of non-missing
    values among training-eligible rows. This matters for Logistic Regression,
    which requires imputation (default 0.40). XGBoost handles missing values
    natively, so it can be called with coverage_thresh=0.0 to use all features.
    """
    df = df.copy()
    df["Sex_F"] = df["Gender"].astype(str).str.contains("נקבה|Female|F", na=False).astype(int)
    demo_feats = ["Age", "Sex_F"]
    # Raw same-visit vitals (VITALS) are excluded: an ablation showed they add no
    # predictive value (AUC 0.761 → 0.762 when removed). What predicts deterioration
    # is the deviation from the patient's baseline, not the absolute reading — so we
    # keep baselines, rolling means, deltas, and drifts, but drop the raw values.
    vital_feats = (
        [f"{v}_baseline"       for v in VITALS] +
        [f"{v}_14d_mean"       for v in VITALS] +
        [f"{v}_30d_mean"       for v in VITALS] +
        [f"{v}_delta_baseline" for v in VITALS] +
        [f"{v}_drift_14v30"    for v in VITALS]
    )
    # Coverage measured on training-eligible rows (Label in {0,1}), not gray zone
    train_pool = df[df["Label"].isin([0, 1])]
    vital_feats = [f for f in vital_feats if train_pool[f].notna().mean() >= coverage_thresh]
    return demo_feats + vital_feats, df


def _training_frame(df):
    """Rows usable for training: stable (0) or imminent (1). Gray zone (NaN) excluded."""
    return df[df["Label"].isin([0, 1])].copy()


def _plot_cv_roc(results, out_path, title):
    """ROC computed from aggregated out-of-fold predictions."""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["steelblue", "tomato", "mediumseagreen", "darkorange"]
    for (name, y_true, y_prob), color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _fit_logreg_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                    solver="lbfgs", random_state=RANDOM_STATE)),
    ])


def _run_cv(df, feats, fit_fn, label_col="Label", group_col="PatID", n_folds=N_FOLDS, name=""):
    """
    Patient-grouped k-fold CV.
    `fit_fn(X_train, y_train)` returns a fitted estimator with predict_proba.
    Returns: (fold_metrics_df, y_true_all, y_prob_all, fitted_fold_models)
    """
    print(f"══ {name} ══  ({len(df):,} rows, {df[group_col].nunique()} patients)")
    gkf = GroupKFold(n_splits=n_folds)
    fold_rows, y_true_all, y_prob_all, models = [], [], [], []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(df, df[label_col], groups=df[group_col])):
        train, test = df.iloc[tr_idx], df.iloc[te_idx]
        model = fit_fn(train[feats], train[label_col])
        models.append(model)

        y_prob = model.predict_proba(test[feats])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        y_true = test[label_col].values

        fold_rows.append({
            "Fold": fold + 1,
            "AUC":       roc_auc_score(y_true, y_prob),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall":    recall_score(y_true, y_pred, zero_division=0),
            "F1":        f1_score(y_true, y_pred, zero_division=0),
            "TestPts":   test[group_col].nunique(),
        })
        y_true_all.extend(y_true)
        y_prob_all.extend(y_prob)

    fm = pd.DataFrame(fold_rows)
    print(fm.round(3).to_string(index=False))
    print(f"   AUC: {fm['AUC'].mean():.3f}±{fm['AUC'].std():.3f} | "
          f"Prec: {fm['Precision'].mean():.3f}±{fm['Precision'].std():.3f} | "
          f"Rec: {fm['Recall'].mean():.3f}±{fm['Recall'].std():.3f} | "
          f"F1: {fm['F1'].mean():.3f}±{fm['F1'].std():.3f}\n")
    return fm, np.array(y_true_all), np.array(y_prob_all), models


def train_baseline(visits_features, figures_dir=FIGURES_DIR):
    """Logistic Regression baseline, patient-grouped 5-fold CV. Two experiments."""
    os.makedirs(figures_dir, exist_ok=True)

    feats, df = _build_feature_set(visits_features)
    df = _training_frame(df)
    df["Label"] = df["Label"].astype(int)
    print(f"Feature set: {len(feats)} features (Age, Sex + {len(feats)-2} vital-based)\n")

    fit_fn = lambda X, y: _fit_logreg_pipeline().fit(X, y)

    fm1, yt1, yp1, _ = _run_cv(df, feats, fit_fn, name="All visits (LR)")
    fm1.to_csv(os.path.join(figures_dir, "baseline_cv_metrics_all.csv"), index=False)

    df_v = df[df["HasAnyVital"] == 1].copy()
    fm2, yt2, yp2, _ = _run_cv(df_v, feats, fit_fn, name="Visits with vitals (LR)")
    fm2.to_csv(os.path.join(figures_dir, "baseline_cv_metrics_vitals.csv"), index=False)

    _plot_cv_roc([("All visits", yt1, yp1), ("Visits with vitals", yt2, yp2)],
                 os.path.join(figures_dir, "fig_baseline_roc_cv.png"),
                 "Logistic Regression Baseline — 5-Fold CV (Aggregated)")
    print(f"✓ Baseline results saved to {figures_dir}")
    return fm1, fm2


def shap_baseline(visits_features, figures_dir=FIGURES_DIR, sample_size=3000):
    """
    SHAP analysis for the Logistic Regression baseline (vitals-only, primary cohort).
    Uses LinearExplainer. Because LR is linear, SHAP importance mirrors the
    standardized coefficients — included for a like-for-like comparison with XGBoost.
    """
    import shap
    os.makedirs(figures_dir, exist_ok=True)

    feats, df = _build_feature_set(visits_features)
    df = _training_frame(df)
    df["Label"] = df["Label"].astype(int)
    dv = df[df["HasAnyVital"] == 1].copy()

    # Fit the full LR pipeline on the primary cohort
    pipe = _fit_logreg_pipeline().fit(dv[feats], dv["Label"])

    # Transform features through impute+scale, then explain the linear model
    imputer = pipe.named_steps["imputer"]
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]

    X_proc = scaler.transform(imputer.transform(dv[feats]))
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X_proc), size=min(sample_size, len(X_proc)), replace=False)
    X_sample = X_proc[sample_idx]

    explainer = shap.LinearExplainer(clf, X_sample)
    shap_values = explainer.shap_values(X_sample)

    # Importance bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feats,
                      plot_type="bar", show=False, max_display=20)
    plt.title("Logistic Regression — SHAP Feature Importance", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_baseline_shap_importance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # Beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feats,
                      show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_baseline_shap_beeswarm.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    importance = pd.DataFrame({
        "Feature": feats,
        "MeanAbsShap": np.abs(shap_values).mean(axis=0),
    }).sort_values("MeanAbsShap", ascending=False)
    importance.to_csv(os.path.join(figures_dir, "baseline_shap_importance.csv"), index=False)

    print("\n══ LOGISTIC REGRESSION — SHAP IMPORTANCE ══")
    print(importance.head(15).round(3).to_string(index=False))
    print(f"\n✓ Baseline SHAP saved to {figures_dir}")
    return importance


# =============================================================================
# 10. XGBOOST MODEL + SHAP + GRAY-ZONE REACH
# =============================================================================

from xgboost import XGBClassifier
import shap


def _fit_xgb(X_train, y_train):
    """XGBoost with tuned hyperparameters (random search, 5-fold CV) and balanced weighting.

    Tuned config favors shallow, heavily-regularized trees with a slow learning rate,
    which reduced overfitting and improved CV AUC from 0.676 to 0.741.
    """
    spw = (1 - y_train.mean()) / y_train.mean()
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=1,
        reg_lambda=5,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def _shap_analysis(model, df_test, feats, figures_dir, sample_size=3000):
    """SHAP importance + beeswarm on a subsample of the test set."""
    
    sample = df_test.sample(min(sample_size, len(df_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample[feats])

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, sample[feats], plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_xgb_shap_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, sample[feats], show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_xgb_shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"   [SHAP] saved figure to: {os.path.abspath(os.path.join(figures_dir, 'fig_xgb_shap_importance.png'))}")

    importance = pd.DataFrame({
        "Feature": feats,
        "MeanAbsShap": np.abs(shap_values).mean(axis=0),
    }).sort_values("MeanAbsShap", ascending=False)
    importance.to_csv(os.path.join(figures_dir, "xgb_shap_importance.csv"), index=False)
    return importance


def _gray_zone_reach(df_all, feats, figures_dir):
    """
    Train XGBoost on stable+imminent (vitals-only), evaluate predicted risk on:
    stable (out-of-fold), imminent (out-of-fold), and gray-zone visits.
    Demonstrates whether the model assigns elevated risk beyond its training window.
    """
    train_pool = df_all[(df_all["Label"].isin([0, 1])) & (df_all["HasAnyVital"] == 1)].copy()
    train_pool["Label"] = train_pool["Label"].astype(int)
    gray = df_all[(df_all["Horizon"] == "gray_zone") & (df_all["HasAnyVital"] == 1)].copy()
    if len(gray) == 0:
        print("No gray-zone visits available; skipping reach analysis.")
        return

    gkf = GroupKFold(n_splits=N_FOLDS)
    train_pool = train_pool.reset_index(drop=True)
    oof = np.zeros(len(train_pool))
    gray_preds = []
    for tr, te in gkf.split(train_pool, train_pool["Label"], groups=train_pool["PatID"]):
        tr_df, te_df = train_pool.iloc[tr], train_pool.iloc[te]
        clf = _fit_xgb(tr_df[feats], tr_df["Label"])
        oof[te] = clf.predict_proba(te_df[feats])[:, 1]
        gray_preds.append(clf.predict_proba(gray[feats])[:, 1])
    gray_prob = np.mean(gray_preds, axis=0)

    p_stable = oof[train_pool["Label"] == 0]
    p_imm = oof[train_pool["Label"] == 1]
    print("══ GRAY-ZONE REACH ══")
    print(f"   Stable      mean prob: {p_stable.mean():.3f} (n={len(p_stable):,})")
    print(f"   Gray zone   mean prob: {gray_prob.mean():.3f} (n={len(gray_prob):,})")
    print(f"   Imminent    mean prob: {p_imm.mean():.3f} (n={len(p_imm):,})")
    auc_gs = roc_auc_score(np.r_[np.zeros(len(p_stable)), np.ones(len(gray_prob))],
                           np.r_[p_stable, gray_prob])
    print(f"   AUC (gray zone vs stable): {auc_gs:.3f}\n")

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 40)
    ax.hist(p_stable, bins=bins, alpha=0.6, density=True, color="steelblue",
            label=f"Stable (mean={p_stable.mean():.2f})")
    ax.hist(gray_prob, bins=bins, alpha=0.6, density=True, color="orange",
            label=f"Gray zone (mean={gray_prob.mean():.2f})")
    ax.hist(p_imm, bins=bins, alpha=0.6, density=True, color="tomato",
            label=f"Imminent (mean={p_imm.mean():.2f})")
    ax.set_xlabel("Predicted probability of deterioration")
    ax.set_ylabel("Density")
    ax.set_title("Model Reach Across Prediction Horizons", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_gray_zone_reach.png"), dpi=150, bbox_inches="tight")
    plt.close()


def train_xgboost(visits_features, visits_features_with_gray=None, figures_dir=FIGURES_DIR):
    """
    XGBoost with patient-grouped 5-fold CV. Two experiments (all visits / vitals-only),
    SHAP analysis, and gray-zone reach analysis.

    NOTE: The "all visits" run benefits partly from a data-collection artifact (visits
    without recorded vitals are over-represented near events). The "vitals-only" run is
    the methodologically clean primary result, matching the patient-self-reporting
    deployment scenario.
    """
    os.makedirs(figures_dir, exist_ok=True)

    feats, df_all = _build_feature_set(visits_features, coverage_thresh=0.0)
    df = _training_frame(df_all)
    df["Label"] = df["Label"].astype(int)
    print(f"Feature set: {len(feats)} features\n")

    fm1, yt1, yp1, models1 = _run_cv(df, feats, _fit_xgb, name="All visits (XGBoost)")
    fm1.to_csv(os.path.join(figures_dir, "xgb_cv_metrics_all.csv"), index=False)

    df_v = df[df["HasAnyVital"] == 1].copy()
    fm2, yt2, yp2, _ = _run_cv(df_v, feats, _fit_xgb, name="Visits with vitals (primary)")
    fm2.to_csv(os.path.join(figures_dir, "xgb_cv_metrics_vitals.csv"), index=False)

    _plot_cv_roc([("All visits", yt1, yp1), ("Visits with vitals (primary)", yt2, yp2)],
                 os.path.join(figures_dir, "fig_xgb_roc_cv.png"),
                 "XGBoost — 5-Fold CV (Aggregated)")

    # SHAP on first fold's model (all-visits)
    gkf = GroupKFold(n_splits=N_FOLDS)
    _, first_te = next(iter(gkf.split(df, df["Label"], groups=df["PatID"])))
    _shap_analysis(models1[0], df.iloc[first_te], feats, figures_dir)

    # Gray-zone reach — needs the with-gray frame (gray-zone rows aren't in visits_features)
    if visits_features_with_gray is not None:
        _, df_gray = _build_feature_set(visits_features_with_gray, coverage_thresh=0.0)
        _gray_zone_reach(df_gray, feats, figures_dir)
    else:
        print("(No with-gray frame provided; skipping gray-zone reach analysis.)")

    print(f"✓ XGBoost results saved to {figures_dir}")
    return fm1, fm2

# =============================================================================
# 11. PATIENT-LEVEL RISK TIERS
# =============================================================================
# Per-visit binary alarms proved clinically impractical due to the low base
# rate (~3.7%): catching 70% of events required flagging ~39% of visits (93%
# false alarms — alarm fatigue). Instead we produce a continuous, time-smoothed
# risk score per patient and map it to outcome-driven tiers.
#
# Two temporal signals (computed from out-of-fold risk scores):
#   - risk_slow: 14-day rolling mean of risk  -> sustained/gradual decline
#   - risk_fast: current raw risk score        -> acute deterioration
#   - risk_combined = max(slow, fast)          -> flags either pattern
#
# A risk-acceleration signal was tested and DROPPED: standalone AUC 0.53, and it
# slightly lowered combined discrimination (0.728 -> 0.725).
#
# Tiers are outcome-driven (each tier has a known deterioration rate):
#   High     score >= 0.6   (~19% deteriorate within 14 days,  ~7% of visits)
#   Moderate 0.5 <= s < 0.6 (~5% deteriorate,                  ~12% of visits)
#   Low      score < 0.5    (~2% deteriorate,                  ~81% of visits)

TIER_HIGH = 0.60
TIER_MODERATE = 0.50


def _oof_risk_scores(df_vitals, feats):
    """Out-of-fold per-visit risk via patient-grouped CV with the tuned XGBoost."""
    df = df_vitals.sort_values(["PatID", "VisitDate"]).reset_index(drop=True)
    gkf = GroupKFold(n_splits=N_FOLDS)
    risk = np.full(len(df), np.nan)
    for tr_idx, te_idx in gkf.split(df, df["Label"], groups=df["PatID"]):
        model = _fit_xgb(df.iloc[tr_idx][feats], df.iloc[tr_idx]["Label"])
        risk[te_idx] = model.predict_proba(df.iloc[te_idx][feats])[:, 1]
    df["risk_raw"] = risk
    return df


def _add_temporal_signals(df):
    """Add risk_slow (14d rolling mean), risk_fast (raw), risk_combined (max)."""
    df = df.sort_values(["PatID", "VisitDate"]).reset_index(drop=True)
    df["risk_slow"] = df.groupby("PatID", group_keys=False).apply(
        lambda g: pd.Series(
            g.set_index("VisitDate")["risk_raw"].rolling("14D").mean().values,
            index=g.index,
        )
    )
    df["risk_fast"] = df["risk_raw"]
    df["risk_combined"] = df[["risk_slow", "risk_fast"]].max(axis=1)
    return df


def _assign_tier(score):
    if score >= TIER_HIGH:
        return "High"
    if score >= TIER_MODERATE:
        return "Moderate"
    return "Low"


def build_risk_tiers(visits_features, coverage_thresh=0.0, figures_dir=FIGURES_DIR):
    """
    Build and validate the patient-level risk-tier system on the primary
    (vitals-only) cohort. Saves a tier-validation figure and prints the
    tier deterioration rates plus patient-level early-warning coverage.
    """
    os.makedirs(figures_dir, exist_ok=True)

    feats, df = _build_feature_set(visits_features, coverage_thresh=0.0)
    df = _training_frame(df)
    df["Label"] = df["Label"].astype(int)
    df["VisitDate"] = pd.to_datetime(df["VisitDate"])

    dv = df[df["HasAnyVital"] == 1].copy()
    dv = _oof_risk_scores(dv, feats)
    dv = _add_temporal_signals(dv)
    dv["Tier"] = dv["risk_combined"].apply(_assign_tier)

    print(f"Out-of-fold AUC (raw risk): "
          f"{roc_auc_score(dv['Label'], dv['risk_raw']):.3f}")

    # Tier validation
    print("\n══ TIER VALIDATION ══")
    for t in ["Low", "Moderate", "High"]:
        band = dv[dv["Tier"] == t]
        print(f"  {t:<9}: {len(band):>6,} visits ({len(band)/len(dv)*100:4.1f}%) | "
              f"deterioration rate {band['Label'].mean()*100:5.1f}%")

    # Patient-level early warning
    det = dv[dv["Label"] == 1]["PatID"].unique()
    reached_high = sum((dv[dv["PatID"] == p]["Tier"] == "High").any() for p in det)
    reached_mod = sum(
        (dv[dv["PatID"] == p]["Tier"] == "Moderate").any()
        and not (dv[dv["PatID"] == p]["Tier"] == "High").any()
        for p in det
    )
    print("\n══ PATIENT-LEVEL EARLY WARNING ══")
    print(f"  Deteriorated patients: {len(det)}")
    print(f"    Reached High:            {reached_high} ({reached_high/len(det)*100:.0f}%)")
    print(f"    Reached Moderate only:   {reached_mod} ({reached_mod/len(det)*100:.0f}%)")
    print(f"    Combined warning:        {(reached_high+reached_mod)/len(det)*100:.0f}%")

    # Figure: tier deterioration rates + score distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    rates = [dv[dv["Tier"] == t]["Label"].mean() * 100 for t in ["Low", "Moderate", "High"]]
    counts = [len(dv[dv["Tier"] == t]) for t in ["Low", "Moderate", "High"]]
    colors = ["mediumseagreen", "orange", "tomato"]
    axes[0].bar(["Low", "Moderate", "High"], rates, color=colors, edgecolor="white", alpha=0.85)
    axes[0].set_ylabel("Deterioration rate within 14 days (%)")
    axes[0].set_title("Deterioration Rate by Risk Tier", fontweight="bold")
    for i, (r, c) in enumerate(zip(rates, counts)):
        axes[0].text(i, r + 0.4, f"{r:.1f}%\n(n={c:,})", ha="center", fontsize=9)

    axes[1].hist(dv[dv["Label"] == 0]["risk_combined"], bins=40, alpha=0.6,
                 density=True, color="steelblue", label="Stable")
    axes[1].hist(dv[dv["Label"] == 1]["risk_combined"], bins=40, alpha=0.6,
                 density=True, color="tomato", label="Imminent")
    axes[1].axvline(TIER_MODERATE, color="orange", linestyle="--",
                    label=f"Moderate ({TIER_MODERATE})")
    axes[1].axvline(TIER_HIGH, color="red", linestyle="--",
                    label=f"High ({TIER_HIGH})")
    axes[1].set_xlabel("Combined risk score")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Risk Score Distribution by Outcome", fontweight="bold")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_risk_tiers.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✓ Risk-tier results saved to {figures_dir}")
    return dv

# =============================================================================
# 12. TRAJECTORY ANALYSIS — how vitals evolve approaching an event
# =============================================================================
# Mechanistic validation: bin pre-event visits by days-to-event and track the
# mean delta-from-baseline of each vital. Confirms the model learns a real,
# physiologically coherent deterioration signal rather than noise.
#
# Observed pattern (textbook HF decompensation):
#   - Systolic BP falls progressively toward the event
#   - Pulse rises (compensatory tachycardia)
#   - Saturation drifts down then drops sharply in the final 1-2 days
#   - Weight rises (fluid retention) starting ~1-2 weeks out

TRAJ_BINS = [0, 2, 5, 8, 11, 14, 18, 22]
TRAJ_LABELS = ["1-2d", "3-5d", "6-8d", "9-11d", "12-14d", "15-18d", "19-21d"]


def trajectory_analysis(visits_features_with_gray, figures_dir=FIGURES_DIR):
    """
    Analyze and plot mean delta-from-baseline of each vital by days-to-event.
    Requires a frame that retains gray-zone rows and DaysToNextEvent
    (i.e. visits_features_with_gray.csv, where Horizon is in
    {stable, imminent, gray_zone}).
    """
    os.makedirs(figures_dir, exist_ok=True)
    df = visits_features_with_gray.copy()
    df["VisitDate"] = pd.to_datetime(df["VisitDate"])

    pre = df[df["Horizon"].isin(["imminent", "gray_zone"])].copy()
    pre["DayBin"] = pd.cut(pre["DaysToNextEvent"], bins=TRAJ_BINS,
                           labels=TRAJ_LABELS, right=True, include_lowest=True)

    # Build the delta-from-baseline trajectory table
    delta_cols = [f"{v}_delta_baseline" for v in VITALS]
    traj = pre.groupby("DayBin")[delta_cols].mean()
    counts = pre.groupby("DayBin").size()

    print("══ MEAN DELTA-FROM-BASELINE BY DAYS-TO-EVENT ══")
    print(traj.round(2).to_string())
    print("\nVisits per bin:")
    print(counts.to_string())

    # Plot: one panel per vital, x-axis ordered far -> near the event
    order = [l for l in TRAJ_LABELS if l in traj.index and not traj.loc[l].isna().all()]
    traj = traj.loc[order[::-1]]  # far (left) -> near (right)

    fig, axes = plt.subplots(1, len(VITALS), figsize=(4 * len(VITALS), 5), sharex=True)
    for ax, v in zip(axes, VITALS):
        ax.plot(traj.index, traj[f"{v}_delta_baseline"], marker="o", lw=2, color="tomato")
        ax.axhline(0, color="gray", linestyle="--", lw=1, label="Patient baseline")
        ax.set_title(v, fontweight="bold")
        ax.set_xlabel("Days before event")
        ax.set_ylabel("Δ from baseline")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Vital Sign Trajectories Approaching Deterioration",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_vital_trajectories.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✓ Trajectory figure saved to {figures_dir}")
    return traj

# =============================================================================
# 13. ABLATIONS — feature contribution tests
# =============================================================================

def current_value_ablation(visits_features, figures_dir=FIGURES_DIR):
    """
    Tests whether predictive power rests on baselines + dynamics, or also on
    the deviation-from-baseline encoding. Compares the adopted model (no raw
    current values) with a stripped variant (no current values AND no deltas).
    Saves SHAP for each.
    """
    import shap
    os.makedirs(figures_dir, exist_ok=True)

    RAW_CURRENT = list(VITALS)
    DELTA_FEATS = [f"{v}_delta_baseline" for v in VITALS]

    feats_full, df = _build_feature_set(visits_features, coverage_thresh=0.0)
    df = _training_frame(df)
    df["Label"] = df["Label"].astype(int)
    dv = df[df["HasAnyVital"] == 1].copy().reset_index(drop=True)

    variants = {
        "no_current":          [f for f in feats_full if f not in RAW_CURRENT],
        "no_current_no_delta": [f for f in feats_full if f not in RAW_CURRENT + DELTA_FEATS],
    }
    name_map = {"no_current": "basic"}

    def _shap_for(feats, tag):
        fname = name_map.get(tag, tag)
        model = _fit_xgb(dv[feats], dv["Label"])
        gkf = GroupKFold(n_splits=N_FOLDS)
        _, te = next(iter(gkf.split(dv, dv["Label"], groups=dv["PatID"])))
        sample = dv.iloc[te].sample(min(3000, len(te)), random_state=RANDOM_STATE)
        sv = shap.TreeExplainer(model).shap_values(sample[feats])

        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, sample[feats], plot_type="bar", show=False, max_display=20)
        plt.title(f"XGBoost SHAP — {fname}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"fig_shap_{fname}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        imp = pd.DataFrame({"Feature": feats, "MeanAbsShap": np.abs(sv).mean(axis=0)}
                           ).sort_values("MeanAbsShap", ascending=False)
        imp.to_csv(os.path.join(figures_dir, f"shap_{fname}.csv"), index=False)

    rows = []
    for tag, feats in variants.items():
        print(f"\n########## XGBoost — {tag} ({len(feats)} features) ##########")
        fm, *_ = _run_cv(dv, feats, _fit_xgb, name=f"XGBoost — {tag}")
        _shap_for(feats, tag)
        rows.append({
            "Variant": tag, "N_features": len(feats),
            "AUC": fm["AUC"].mean(), "AUC_std": fm["AUC"].std(),
            "Precision": fm["Precision"].mean(), "Recall": fm["Recall"].mean(),
            "F1": fm["F1"].mean(),
        })

    summary = pd.DataFrame(rows)
    print("\n══════════ CURRENT-VALUE ABLATION SUMMARY ══════════")
    print(summary.round(3).to_string(index=False))
    base = summary[summary["Variant"] == "no_current"]["AUC"].values[0]
    print("\n── AUC change vs 'no_current' (the adopted 'basic' model) ──")
    for _, r in summary.iterrows():
        print(f"  {r['Variant']:<22} AUC {r['AUC']:.3f}  (Δ = {r['AUC']-base:+.3f})")
    print(f"\n✓ SHAP figures saved → {figures_dir}")
    return summary


def baseline_only_ablation(visits_features, figures_dir=FIGURES_DIR):
    import shap
    os.makedirs(figures_dir, exist_ok=True)

    # Baseline-only feature set
    BASELINE_FEATS = [f"{v}_baseline" for v in VITALS]
    feats_baseline = ["Age", "Sex_F"] + BASELINE_FEATS
    print(f"Baseline-only feature set ({len(feats_baseline)}): {feats_baseline}\n")

    # Prepare training frame (same cohort, same CV, same Sex_F encoding)
    _, df = _build_feature_set(visits_features, coverage_thresh=0.0)
    df = _training_frame(df)
    df["Label"] = df["Label"].astype(int)
    dv = df[df["HasAnyVital"] == 1].copy().reset_index(drop=True)
    print(f"Cohort: {len(dv):,} vitals-only visits, {dv['PatID'].nunique()} patients\n")

    def _shap_for(model_name, model, sample_X, sample_feats, tag, explainer_cls):
        sv = explainer_cls(model, sample_X if explainer_cls is shap.LinearExplainer
                                  else None).shap_values(sample_X)
        plt.figure(figsize=(9, 6))
        shap.summary_plot(sv, sample_X, feature_names=sample_feats,
                          plot_type="bar", show=False, max_display=15)
        plt.title(f"{model_name} — SHAP, baseline-only features", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"fig_shap_baseline_only_{tag}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        imp = pd.DataFrame({"Feature": sample_feats,
                            "MeanAbsShap": np.abs(sv).mean(axis=0)}
                          ).sort_values("MeanAbsShap", ascending=False)
        imp.to_csv(os.path.join(figures_dir, f"shap_baseline_only_{tag}.csv"), index=False)

    rows = []

    # ── XGBoost ─────────────────────────────────────────────────────────────
    print("########## XGBoost — baseline-only ##########")
    fm_xgb, *_ = _run_cv(dv, feats_baseline, _fit_xgb, name="XGBoost — baseline-only")
    # Fit once on full cohort for SHAP
    model_xgb = _fit_xgb(dv[feats_baseline], dv["Label"])
    gkf = GroupKFold(n_splits=N_FOLDS)
    _, te = next(iter(gkf.split(dv, dv["Label"], groups=dv["PatID"])))
    sample = dv.iloc[te].sample(min(3000, len(te)), random_state=RANDOM_STATE)
    _shap_for("XGBoost", model_xgb, sample[feats_baseline], feats_baseline,
              "xgb", shap.TreeExplainer)
    rows.append({"Model": "XGBoost", "N_features": len(feats_baseline),
                 "AUC": fm_xgb["AUC"].mean(), "AUC_std": fm_xgb["AUC"].std(),
                 "Precision": fm_xgb["Precision"].mean(),
                 "Recall": fm_xgb["Recall"].mean(),
                 "F1": fm_xgb["F1"].mean()})

    # ── Logistic Regression ─────────────────────────────────────────────────
    print("\n########## Logistic Regression — baseline-only ##########")
    fit_lr = lambda X, y: _fit_logreg_pipeline().fit(X, y)
    fm_lr, *_ = _run_cv(dv, feats_baseline, fit_lr, name="LR — baseline-only")
    pipe_lr = fit_lr(dv[feats_baseline], dv["Label"])
    # For LR SHAP: explain the inner linear model on imputed+scaled data
    X_proc = pipe_lr.named_steps["scaler"].transform(
        pipe_lr.named_steps["imputer"].transform(dv[feats_baseline]))
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X_proc), size=min(3000, len(X_proc)), replace=False)
    X_sample = X_proc[sample_idx]
    explainer = shap.LinearExplainer(pipe_lr.named_steps["clf"], X_sample)
    sv = explainer.shap_values(X_sample)
    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, X_sample, feature_names=feats_baseline,
                      plot_type="bar", show=False, max_display=15)
    plt.title("Logistic Regression — SHAP, baseline-only features", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_shap_baseline_only_lr.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    pd.DataFrame({"Feature": feats_baseline,
                  "MeanAbsShap": np.abs(sv).mean(axis=0)}
                ).sort_values("MeanAbsShap", ascending=False
                ).to_csv(os.path.join(figures_dir, "shap_baseline_only_lr.csv"), index=False)

    rows.append({"Model": "Logistic Regression", "N_features": len(feats_baseline),
                 "AUC": fm_lr["AUC"].mean(), "AUC_std": fm_lr["AUC"].std(),
                 "Precision": fm_lr["Precision"].mean(),
                 "Recall": fm_lr["Recall"].mean(),
                 "F1": fm_lr["F1"].mean()})

    # ── Summary ─────────────────────────────────────────────────────────────
    summary = pd.DataFrame(rows)
    print("\n\n══════════ BASELINE-ONLY ABLATION SUMMARY ══════════")
    print(summary.round(3).to_string(index=False))

    print("\n── Compare to your current 'basic' model (AUC 0.762 for XGBoost, 0.632 for LR) ──")
    ref = {"XGBoost": 0.762, "Logistic Regression": 0.632}
    for _, r in summary.iterrows():
        delta = r["AUC"] - ref[r["Model"]]
        print(f"  {r['Model']:<22} {r['AUC']:.3f}  (Δ vs basic = {delta:+.3f})")

    print(f"\n✓ SHAP figures saved: fig_shap_baseline_only_xgb.png, "
          f"fig_shap_baseline_only_lr.png  →  {figures_dir}")
    return summary


# =============================================================================
# 14. NESTED CV + CALIBRATION
# =============================================================================

def nested_cv_and_calibration(visits_features, figures_dir=FIGURES_DIR):
    from sklearn.model_selection import GroupKFold
    from sklearn.calibration import calibration_curve
    os.makedirs(figures_dir, exist_ok=True)

    # ── 1. Prepare cohort (same as primary model) ────────────────────────────
    feats, df = _build_feature_set(visits_features, coverage_thresh=0.0)
    df = _training_frame(df)
    df["Label"] = df["Label"].astype(int)
    dv = df[df["HasAnyVital"] == 1].copy().reset_index(drop=True)
    print(f"Cohort: {len(dv):,} vitals-only visits, {dv['PatID'].nunique()} patients\n")

    # ── 2. Nested CV ─────────────────────────────────────────────────────────
    # Outer loop: 5 folds, fully held-out evaluation.
    # Inner loop: 3 folds, small grid search near the locked params.
    INNER_GRID = [
        # Locked params (the reference point)
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.01,
         "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 1, "reg_lambda": 5},
        # Small perturbations around it
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.01,
         "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 1, "reg_lambda": 5},
        {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.01,
         "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 1, "reg_lambda": 5},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.01,
         "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 1, "reg_lambda": 5},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03,
         "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 1, "reg_lambda": 5},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.01,
         "subsample": 1.0, "colsample_bytree": 0.7, "min_child_weight": 1, "reg_lambda": 5},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.01,
         "subsample": 0.8, "colsample_bytree": 1.0, "min_child_weight": 1, "reg_lambda": 5},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.01,
         "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 3, "reg_lambda": 5},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.01,
         "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 1, "reg_lambda": 3},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.01,
         "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 1, "reg_lambda": 10},
    ]

    def _fit_with_params(X, y, params):
        from xgboost import XGBClassifier
        spw = (1 - y.mean()) / y.mean()
        clf = XGBClassifier(scale_pos_weight=spw, random_state=RANDOM_STATE,
                            eval_metric="logloss", tree_method="hist",
                            n_jobs=-1, **params)
        clf.fit(X, y)
        return clf

    print("══ NESTED CV ══")
    outer = GroupKFold(n_splits=N_FOLDS)
    outer_aucs, outer_probs, outer_truth, chosen_params = [], [], [], []

    for outer_fold, (tr_idx, te_idx) in enumerate(
        outer.split(dv, dv["Label"], groups=dv["PatID"]), 1
    ):
        train, test = dv.iloc[tr_idx], dv.iloc[te_idx]

        # Inner CV: find best params on train only
        inner = GroupKFold(n_splits=3)
        best_params, best_auc = None, -1
        for params in INNER_GRID:
            inner_aucs = []
            for inner_tr, inner_te in inner.split(train, train["Label"], groups=train["PatID"]):
                itr, ite = train.iloc[inner_tr], train.iloc[inner_te]
                clf = _fit_with_params(itr[feats], itr["Label"], params)
                p = clf.predict_proba(ite[feats])[:, 1]
                inner_aucs.append(roc_auc_score(ite["Label"], p))
            mean_auc = np.mean(inner_aucs)
            if mean_auc > best_auc:
                best_auc, best_params = mean_auc, params

        # Fit best params on full outer-train, evaluate on held-out outer-test
        model = _fit_with_params(train[feats], train["Label"], best_params)
        prob = model.predict_proba(test[feats])[:, 1]
        auc = roc_auc_score(test["Label"], prob)
        outer_aucs.append(auc)
        outer_probs.extend(prob); outer_truth.extend(test["Label"].values)
        chosen_params.append(best_params)
        print(f"  Outer fold {outer_fold}: AUC = {auc:.3f}   "
              f"(inner-best AUC = {best_auc:.3f}, params: depth={best_params['max_depth']}, "
              f"n_est={best_params['n_estimators']}, lr={best_params['learning_rate']})")

    outer_aucs = np.array(outer_aucs)
    print(f"\n  Nested-CV AUC: {outer_aucs.mean():.3f} ± {outer_aucs.std():.3f}")
    print(f"  (compare to non-nested CV AUC reported earlier — gap indicates optimism)")

    # ── 3. Calibration (overall) ─────────────────────────────────────────────
    y_true = np.array(outer_truth)
    y_prob = np.array(outer_probs)

    print("\n══ CALIBRATION (overall) ══")
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    from sklearn.metrics import brier_score_loss
    brier = brier_score_loss(y_true, y_prob)
    print(f"  Brier score: {brier:.4f}  (lower is better; 0 = perfect, 0.25 = random)")
    print(f"  {'Predicted prob':>15}  →  {'Observed rate':>14}")
    for mp, fp in zip(mean_pred, frac_pos):
        print(f"  {mp:>15.3f}  →  {fp:>14.3f}")

    # ── 4. Calibration by risk tier ──────────────────────────────────────────
    print("\n══ CALIBRATION BY RISK TIER ══")
    # Reconstruct slow/fast/combined risk on the same out-of-fold predictions
    # (use the out-of-fold predictions from nested CV)
    dv_eval = dv.copy()
    dv_eval["VisitDate"] = pd.to_datetime(dv_eval["VisitDate"])
    dv_eval["risk_raw"] = np.nan
    # Map out-of-fold predictions back to dv positions (we kept te_idx order via extend)
    # Easier: rebuild OOF using nested predictions via the same outer split order.
    outer2 = GroupKFold(n_splits=N_FOLDS)
    pos = 0
    for tr_idx, te_idx in outer2.split(dv, dv["Label"], groups=dv["PatID"]):
        n = len(te_idx)
        dv_eval.iloc[te_idx, dv_eval.columns.get_loc("risk_raw")] = y_prob[pos:pos+n]
        pos += n
    dv_eval = _add_temporal_signals(dv_eval)
    dv_eval["Tier"] = dv_eval["risk_combined"].apply(_assign_tier)

    rows = []
    for tier in ["Low", "Moderate", "High"]:
        band = dv_eval[dv_eval["Tier"] == tier]
        if len(band) == 0: continue
        pred_mean = band["risk_combined"].mean()
        obs_rate = band["Label"].mean()
        rows.append({
            "Tier": tier,
            "N_visits": len(band),
            "Mean_predicted": pred_mean,
            "Observed_rate": obs_rate,
            "Gap": pred_mean - obs_rate,
        })
        print(f"  {tier:<10}: n={len(band):>6,}  "
              f"predicted={pred_mean:.3f}  observed={obs_rate:.3f}  "
              f"gap={pred_mean - obs_rate:+.3f}")
    tier_cal = pd.DataFrame(rows)

    # ── 5. Plot calibration curve ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, marker="o", color="#0E7C7B", lw=2,
            label=f"Model (Brier = {brier:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed deterioration rate")
    ax.set_title("Calibration Curve — Nested CV", fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_calibration.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── 6. Save numeric outputs ──────────────────────────────────────────────
    pd.DataFrame({
        "Fold": list(range(1, len(outer_aucs) + 1)),
        "AUC": outer_aucs,
    }).to_csv(os.path.join(figures_dir, "nested_cv_aucs.csv"), index=False)
    tier_cal.to_csv(os.path.join(figures_dir, "calibration_by_tier.csv"), index=False)
    pd.DataFrame({"mean_predicted": mean_pred,
                  "observed_rate": frac_pos}).to_csv(
        os.path.join(figures_dir, "calibration_overall.csv"), index=False)

    print(f"\n✓ Saved: fig_calibration.png, nested_cv_aucs.csv, "
          f"calibration_overall.csv, calibration_by_tier.csv  →  {figures_dir}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # ── Pipeline (always runs) ────────────────────────────────────────────────
    patient_level, visits_unified, visits_with_gray_raw = build_unified_dataset()

    print("\nEngineering features...")
    visits_features = engineer_features(visits_unified)
    visits_features.to_csv(os.path.join(PROCESSED_DIR, "visits_features.csv"), index=False)
    print(f"  Train file: {visits_features.shape[0]:,} rows × {visits_features.shape[1]} columns")

    visits_features_gray = engineer_features(visits_with_gray_raw)
    visits_features_gray.to_csv(os.path.join(PROCESSED_DIR, "visits_features_with_gray.csv"), index=False)
    print(f"  With-gray file: {visits_features_gray.shape[0]:,} rows")

    # ── Feature diagnostics ───────────────────────────────────────────────────
    if RUN_FEATURE_EVAL:
        print("\nEvaluating features...")
        evaluate_features(visits_features)

    # ── Baseline model ────────────────────────────────────────────────────────
    if RUN_BASELINE_LR:
        print("\nTraining baseline (Logistic Regression, 5-fold CV)...")
        train_baseline(visits_features)
        print("\nSHAP analysis for baseline (Logistic Regression)...")
        shap_baseline(visits_features)

    # ── Main model ────────────────────────────────────────────────────────────
    if RUN_XGBOOST:
        print("\nTraining XGBoost (5-fold CV)...")
        train_xgboost(visits_features, visits_features_with_gray=visits_features_gray)

    # ── Patient-level risk tiers ──────────────────────────────────────────────
    if RUN_RISK_TIERS:
        print("\nBuilding patient-level risk tiers...")
        build_risk_tiers(visits_features)

    # ── Trajectory analysis ───────────────────────────────────────────────────
    if RUN_TRAJECTORY:
        print("\nTrajectory analysis...")
        trajectory_analysis(visits_features_gray)

    # ── Feature ablations ─────────────────────────────────────────────────────
    if RUN_ABLATIONS:
        print("\nCurrent-value ablation...")
        current_value_ablation(visits_features)
        print("\nBaseline-only ablation...")
        baseline_only_ablation(visits_features)

    # ── Nested CV + calibration (slow) ────────────────────────────────────────
    if RUN_NESTED_CV:
        print("\nNested CV + calibration...")
        nested_cv_and_calibration(visits_features)

    print("\n✓ Pipeline complete.")
