"""
EXHEART - Disparity-source attribution utilities.

Reusable functions for the cross-instrument fairness-disparity attribution
analysis (Notebook 05). Mirrors the conventions used in utils/fairness.py.

Author: Md Anas Biswas, University of Portsmouth.
GitHub: https://github.com/anasbiswas1/exheart-research
"""
import numpy as np
import pandas as pd


def tpr_at_threshold(y_true, y_prob, threshold):
    """Sensitivity (recall) at a given operating threshold."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    pos = (y_true == 1)
    if pos.sum() == 0:
        return np.nan
    return float(((y_pred == 1) & pos).sum() / pos.sum())


def group_tpr(y_true, y_prob, groups, threshold):
    """Return {group_level: TPR} for each level in `groups`."""
    g = np.asarray(groups)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    out = {}
    for lvl in pd.unique(g):
        m = (g == lvl)
        if m.sum() == 0:
            continue
        out[lvl] = tpr_at_threshold(y_true[m], y_prob[m], threshold)
    return out


def tpr_gap(y_true, y_prob, groups, threshold, min_n=30):
    """Max-minus-min TPR gap across groups with at least `min_n` samples."""
    g = np.asarray(groups)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    tprs = []
    for lvl in pd.unique(g):
        m = (g == lvl)
        if m.sum() < min_n:
            continue
        t = tpr_at_threshold(y_true[m], y_prob[m], threshold)
        if not np.isnan(t):
            tprs.append(t)
    if len(tprs) < 2:
        return np.nan
    return float(max(tprs) - min(tprs))


def relative_gap_reduction(gap_survey, gap_clinical):
    """Attribution score: the fraction of the survey disparity that disappears
    on the objective-measurement instrument.

    ~1.0  => measurement-induced (collapses on clinical data)
    ~0.0  => substantive (persists on clinical data)
    """
    if gap_survey is None or np.isnan(gap_survey) or gap_survey == 0:
        return np.nan
    return float((gap_survey - gap_clinical) / gap_survey)


def bootstrap_metric_ci(func, n, B=1000, seed=42, alpha=0.05):
    """Generic non-parametric bootstrap over an index 0..n-1.

    `func(idx)` must accept an index array and return a scalar metric.
    Returns (point_estimate, ci_low, ci_high, full_bootstrap_distribution).
    """
    rng = np.random.default_rng(seed)
    point = func(np.arange(n))
    stats = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        stats[b] = func(idx)
    lo = float(np.nanpercentile(stats, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(stats, 100 * (1 - alpha / 2)))
    return float(point), lo, hi, stats


def binarize_cardio(df):
    """Convert measured Cardio features into BRFSS-style coarse flags using
    standard clinical cutoffs, so that ONLY measurement granularity changes
    (same patients, same split). Used for the measured-vs-survey-style test.

    Expects raw-ish Cardio columns: ap_hi, ap_lo, cholesterol, gluc, weight,
    height, gender, smoke, alco, active, and either `age_years` or `age` (days).
    Cutoffs are illustrative and can be edited to match a target survey coding.
    """
    out = pd.DataFrame(index=df.index)
    # Blood pressure -> hypertension flag (ACC/AHA stage-2 style >=140/90)
    out["HighBP_flag"] = ((df["ap_hi"] >= 140) | (df["ap_lo"] >= 90)).astype(int)
    # Cholesterol category (1 normal, 2 above, 3 well-above) -> high flag
    out["HighChol_flag"] = (df["cholesterol"] >= 2).astype(int)
    # Glucose category -> high flag
    out["HighGluc_flag"] = (df["gluc"] >= 2).astype(int)
    # BMI -> coded bands (mirror BRFSS-style coarse coding)
    bmi = df["weight"] / (df["height"] / 100.0) ** 2
    out["BMI_band"] = pd.cut(bmi, bins=[0, 18.5, 25, 30, 35, np.inf],
                             labels=[0, 1, 2, 3, 4]).astype(float)
    # Age years -> 5-year bands (coarsen to a BRFSS-like ordinal scale)
    age_years = df["age_years"] if "age_years" in df.columns else df["age"] / 365.25
    out["Age_band"] = pd.cut(age_years, bins=range(20, 86, 5),
                             labels=False, include_lowest=True).astype(float)
    # Pass through the variables that are already categorical/binary in Cardio
    for c in ["gender", "smoke", "alco", "active"]:
        out[c] = df[c].values
    return out
