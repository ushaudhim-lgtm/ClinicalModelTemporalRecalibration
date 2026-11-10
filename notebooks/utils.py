# common helper functions used by all notebooks in this project
# student: Usha Oudeacoumar (20057009)

import os
import re
import json
import urllib.request

import numpy as np
import pandas as pd
import pdfplumber
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

# folders used everywhere
RAW_DIR = "../data/raw"
META_DIR = "../data/metadata"
PROC_DIR = "../data/processed"
CONFIG_DIR = "../configs"
MODELS_DIR = "../models"
OUT_TABLES = "../outputs/tables"
OUT_FIGS = "../outputs/figures"
OUT_REPORTS = "../outputs/reports"

STUDY_YEARS = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
DEV_YEARS = [2011, 2012, 2013, 2014]
EVAL_YEARS = [2015, 2016, 2017, 2018, 2019]

# file names on the cdc ftp site, these are not the same every year so just hardcode them
DATA_FILES = {
    2011: "ed2011.zip",
    2012: "ED2012.zip",
    2013: "ED2013.zip",
    2014: "ED2014.zip",
    2015: "ED2015.zip",
    2016: "ed2016.zip",
    2017: "ED2017.zip",
    2018: "ED2018.zip",
    2019: "ED2019.zip",
}

DOC_FILES = {
    2011: "doc11.pdf",
    2012: "doc12_ed.pdf",
    2013: "doc13_ed.pdf",
    2014: "doc14_ed.pdf",
    2015: "doc15_ed.pdf",
    2016: "doc16_ed.pdf",
    2017: "doc17_ed-508.pdf",
    2018: "doc18-ed-508.pdf",
    2019: "doc19-ed-508.pdf",
}

DATA_BASE_URL = "https://ftp.cdc.gov/pub/Health_Statistics/nchs/Datasets/NHAMCS"
DOC_BASE_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS"


def download_file(url, out_path):
    """download a single file if it is not already there"""
    if os.path.exists(out_path):
        print("already have", out_path)
        return
    print("downloading", url)
    urllib.request.urlretrieve(url, out_path)


def raw_file_path(yr):
    """path of the unzipped fixed width file for a year, always upper case EDyyyy"""
    return os.path.join(RAW_DIR, "ED" + str(yr))


def doc_file_path(yr):
    """path of the codebook pdf for a year"""
    return os.path.join(META_DIR, DOC_FILES[yr])


# ---------------- codebook parsing ----------------
# the codebook pdf has a table like: item_no field_len location [NAME] description...
# we need this to know where each column starts/ends in the raw fixed width file

item_line_pat = re.compile(r"^(\d+)\s+(\d+)\s+(\d+(?:-\d+)?)\s+\[([^\]]+)\]")


def parse_codebook(pdf_path):
    """read the codebook pdf and return a list of (name, start_col, end_col)"""

    with pdfplumber.open(pdf_path) as pdf:
        start_page = None
        for i, pg in enumerate(pdf.pages):
            txt = pg.extract_text() or ""
            if "LOCATION" in txt and "LENGTH" in txt:
                start_page = i
                break

        end_page = len(pdf.pages) - 1
        zero_count = 0
        for i in range(start_page, len(pdf.pages)):
            txt = pdf.pages[i].extract_text() or ""
            if "OUTPATIENT DEPARTMENT MICRO-DATA FILE" in txt.upper():
                end_page = i - 1
                break
            lines = txt.split("\n")
            has_item = any(item_line_pat.match(l) for l in lines)
            if has_item:
                zero_count = 0
            else:
                zero_count += 1
                if zero_count >= 2:
                    end_page = i - 2
                    break

        all_lines = []
        for i in range(start_page, end_page + 1):
            txt = pdf.pages[i].extract_text() or ""
            all_lines += txt.split("\n")

    out = []
    for l in all_lines:
        m = item_line_pat.match(l)
        if not m:
            continue
        item_no, length_str, loc, name = m.groups()
        length = int(length_str)
        if "-" in loc:
            s, e = loc.split("-")
            s, e = int(s), int(e)
        else:
            s = e = int(loc)
        implied_len = e - s + 1
        # pdf has typos sometimes, if the gap is silly just trust field length instead
        if implied_len != length and (implied_len <= 0 or implied_len > 20):
            e = s + length - 1
        out.append((name.strip(), s, e))
    return out


def load_year_raw(yr):
    """load one year of nhamcs ed data as a dataframe of strings, using the codebook layout"""
    layout = parse_codebook(doc_file_path(yr))
    names = [x[0] for x in layout]
    colspecs = [(x[1] - 1, x[2]) for x in layout]
    df = pd.read_fwf(raw_file_path(yr), colspecs=colspecs, names=names, dtype=str)
    df["yr"] = yr
    return df


# ---------------- outcome ----------------
ADMIT_COLS = ["ADMITHOS", "OBSHOS"]
TRANSFER_COLS = ["TRANNH", "TRANPSYC", "TRANOTH"]
DEATH_COLS = ["DOA", "DIEDED"]


def add_admit_flag(df):
    """add admit column (1/0) based on the admithos/obshos checkbox items"""
    admit = pd.Series(0, index=df.index)
    for c in ADMIT_COLS:
        admit = admit | (pd.to_numeric(df[c], errors="coerce") == 1).astype(int)
    df = df.copy()
    df["admit"] = admit.astype(int)
    return df


def filter_adults(df):
    """keep only age 18 and up"""
    age = pd.to_numeric(df["AGE"], errors="coerce")
    return df[age >= 18].copy()


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------- metrics (discrimination + calibration) ----------------

def calc_cal_intercept_slope(y_true, y_prob):
    """fit logit(y) ~ a + b*logit(p), a=0 and b=1 would be perfect calibration"""
    p = np.clip(y_prob, 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=np.inf)
    lr.fit(logit_p, y_true)
    intercept = lr.intercept_[0]
    slope = lr.coef_[0][0]
    return intercept, slope


def calc_ece(y_true, y_prob, n_bins=10):
    """expected calibration error, bins predictions 0-1 and compares to actual rate"""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        if in_bin.sum() == 0:
            continue
        avg_pred = y_prob[in_bin].mean()
        avg_true = y_true[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(avg_pred - avg_true)
    return ece


def get_all_metrics(y_true, y_prob):
    """one call that gives back auroc, brier, cal intercept/slope and ece"""
    auroc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    cal_intercept, cal_slope = calc_cal_intercept_slope(y_true, y_prob)
    ece = calc_ece(y_true, y_prob)
    return {
        "auroc": auroc,
        "brier": brier,
        "cal_intercept": cal_intercept,
        "cal_slope": cal_slope,
        "ece": ece,
    }
