"""
Conformalized survival analysis (CQR) for the SigBERT-MIMIC pipeline.

Python translation of the *conformalized quantile regression* (CQR) subset of the R
package in ``conformal_survival/`` (Sesia & Svetnik, "Doubly Robust Conformalized
Survival Analysis with Right-Censored Data", ICML 2025).

What is implemented (scope: CQR baseline + marginal Kaplan-Meier censoring):

- ``CoxSurvivalWrapper``       -> R ``CoxphModelWrapper$predict`` + default ``predict_quantiles``
- ``predict_CQR``              -> R ``predict_CQR``            (utils_conformal.R)
- ``fit_survival_km``          -> marginal survival KM ``survfit(Surv(time, status) ~ 1)``
- ``sample_km_conditional`` /
  ``impute_km_decensored`` /
  ``predict_decensoring``      -> R ``predict_decensoring``    (utils_decensoring.R, Qi et al.)
- ``evaluate_bounds``          -> R ``evaluate_bounds``        (utils_conformal.R, MIMIC-adapted)
- ``run_cqr_conformal``        -> convenience driver for the notebook pipeline outputs.

Note on the "marginal Kaplan-Meier" nuisance: within this CQR scope the only KM needed is the
*survival-time* KM, used by the de-censoring step to impute event times for censored
calibration points (conditioning on ``T > C``, following Qi et al.). A *censoring*-distribution
KM would only be required by the weighted methods (Candes/Gui/DR-COSARC), which are out of
scope here.

Each conformal method returns a per-patient *lower predictive bound* (LPB) ``L(x)`` on the
survival time such that ``P(T >= L(x)) >= 1 - alpha``.

The module intentionally depends only on numpy / pandas / lifelines (no scikit-learn or
skglm), so it is unaffected by the notebook's current scikit-learn import issue and can be
imported / validated on its own.

The survival-curve convention matches the rest of the notebook
(``brier_score_ipcw_with_cph`` in ``metrics_plot_results_pkg.py``):

    S(t | x) = baseline_survival(t) ** exp(risk_score)

where ``baseline_survival`` linearly interpolates ``cph.baseline_survival_`` and is clamped
to the first / last value outside the observed time grid.
"""

from __future__ import annotations
from statistics import NormalDist
from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "CoxSurvivalWrapper",
    "predict_CQR",
    "fit_survival_km",
    "sample_km_conditional",
    "impute_km_decensored",
    "predict_decensoring",
    "evaluate_bounds",
    "run_cqr_conformal",
    "plot_conformal_results",
    "_jackknife_mean_ci",
    "summarize_conformal",
    
]


# --------------------------------------------------------------------------------------
# Type-1 quantile (R `quantile(..., type = 1)` == inverse of the empirical CDF)
# --------------------------------------------------------------------------------------
def _quantile_type1(x: np.ndarray, p: float) -> float:
    """R ``quantile(x, p, type = 1)`` -- inverse of the empirical CDF.

    numpy's ``method="inverted_cdf"`` implements Hyndman & Fan type 1 (== R type 1);
    a manual order-statistic fallback is kept for older numpy.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    try:
        return float(np.quantile(x, p, method="inverted_cdf"))
    except TypeError:  # numpy < 1.22
        xs = np.sort(x)
        n = xs.size
        h = n * p
        j = int(np.floor(h))
        gamma = 0.0 if (h - j) == 0 else 1.0
        pos = int(j + gamma) - 1  # 1-indexed -> 0-indexed
        pos = min(max(pos, 0), n - 1)
        return float(xs[pos])


# --------------------------------------------------------------------------------------
# Survival model wrapper around the already-fitted lifelines Cox model
# --------------------------------------------------------------------------------------
class CoxSurvivalWrapper:
    """Wrap a fitted Cox model's baseline survival to expose survival curves and quantiles.

    Only the per-patient ``risk_score`` (the Cox linear predictor ``X @ w``) is needed at
    prediction time, so no model re-fit is required: this consumes
    ``cph.baseline_survival_`` and the ``risk_score`` column already produced by the
    SigBERT pipeline.

    Parameters
    ----------
    baseline_survival : pandas.Series
        Baseline survival function S0(t), indexed by time (e.g.
        ``cph.baseline_survival_.squeeze()``).
    """

    def __init__(self, baseline_survival: pd.Series):
        times = np.asarray(baseline_survival.index, dtype=float)
        surv = np.asarray(baseline_survival.values, dtype=float).ravel()
        order = np.argsort(times)
        self.times = times[order]
        self.surv = np.clip(surv[order], 0.0, 1.0)
        self.t_min = float(self.times[0])
        self.t_max = float(self.times[-1])

    def _baseline_survival_at(self, t: np.ndarray) -> np.ndarray:
        # np.interp clamps to the endpoint values outside the grid, exactly like the
        # baseline_survival() helper in brier_score_ipcw_with_cph.
        return np.interp(np.asarray(t, dtype=float), self.times, self.surv)

    def predict(self, risk_scores: np.ndarray, time_points: np.ndarray) -> np.ndarray:
        """Return the survival probability matrix S(t | x) of shape (n_patients, n_times)."""
        risk_scores = np.asarray(risk_scores, dtype=float).ravel()
        time_points = np.asarray(time_points, dtype=float).ravel()
        s0 = self._baseline_survival_at(time_points)          # (n_times,)
        expr = np.exp(risk_scores)                            # (n_patients,)
        with np.errstate(invalid="ignore"):
            return np.power(s0[None, :], expr[:, None])       # (n_patients, n_times)

    def predict_quantiles(self, risk_scores: np.ndarray, probs: Sequence[float]) -> np.ndarray:
        """Survival-time quantiles by inverting the survival curve.

        For probability ``p`` this returns the time ``t`` at which ``S(t | x) = 1 - p``
        (the p-th quantile of the survival time). Mirrors the R default
        ``SurvivalModelWrapper$predict_quantiles``: pad the curve with ``(t=0, S=1)`` and
        ``(t=t_max+1, S=0)`` and linearly interpolate (``approx(..., rule = 2)``).

        Returns an array of shape ``(n_patients, len(probs))``.
        """
        risk_scores = np.asarray(risk_scores, dtype=float).ravel()
        probs = np.atleast_1d(np.asarray(probs, dtype=float))
        targets = 1.0 - probs  # target survival probabilities

        # Survival curves on the baseline grid, padded at both ends.
        S = self.predict(risk_scores, self.times)             # (n, T)
        n = S.shape[0]
        times_pad = np.concatenate(([0.0], self.times, [self.t_max + 1.0]))
        S_pad = np.concatenate([np.ones((n, 1)), S, np.zeros((n, 1))], axis=1)

        # Survival decreases with time; reverse so the interpolation x-axis (survival)
        # is ascending, as required by np.interp.
        t_rev = times_pad[::-1]
        out = np.empty((n, targets.size), dtype=float)
        for i in range(n):
            out[i, :] = np.interp(targets, S_pad[i, ::-1], t_rev)
        return out


# --------------------------------------------------------------------------------------
# Conformalized Quantile Regression (split conformal)
# --------------------------------------------------------------------------------------
def _as_arrays(df: pd.DataFrame):
    return (
        df["risk_score"].to_numpy(dtype=float),
        df["time"].to_numpy(dtype=float),
    )


def predict_CQR(
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    wrapper: CoxSurvivalWrapper,
    alpha: float,
) -> np.ndarray:
    """Conformalized quantile regression lower bounds (R ``predict_CQR``).

    Parameters
    ----------
    cal_df, test_df : DataFrame
        Must contain ``risk_score`` and ``time`` columns (``time`` = observed Y = min(T, C)).
    wrapper : CoxSurvivalWrapper
    alpha : float
        Miscoverage level; the returned bound satisfies ``P(T >= L) >= 1 - alpha``.

    Returns
    -------
    numpy.ndarray
        Per-patient lower bound for ``test_df`` (same length / order as ``test_df``).
    """
    probs = [alpha]

    rs_cal, y_cal = _as_arrays(cal_df)
    pred_cal = wrapper.predict_quantiles(rs_cal, probs)[:, 0]
    # Defensive: replace NaNs with a large number (as in R). Should not occur here.
    max_ycal = np.nanmax(y_cal)
    pred_cal = np.where(np.isnan(pred_cal), max_ycal, pred_cal)

    scores = pred_cal - y_cal
    n = scores.size
    percentile = (1.0 - alpha) * (1.0 + 1.0 / n)
    calibration = _quantile_type1(scores, percentile) if percentile < 1 else np.inf

    rs_test, _ = _as_arrays(test_df)
    pred_test = wrapper.predict_quantiles(rs_test, probs)[:, 0]
    return np.maximum(0.0, pred_test - calibration)


# --------------------------------------------------------------------------------------
# Marginal survival Kaplan-Meier + KM decensoring (Qi et al.)
# --------------------------------------------------------------------------------------
class _KMCurve:
    """Minimal container mirroring R ``km_fit$time`` / ``km_fit$surv``."""

    def __init__(self, time: np.ndarray, surv: np.ndarray):
        self.time = np.asarray(time, dtype=float)
        self.surv = np.asarray(surv, dtype=float)


def fit_survival_km(df: pd.DataFrame) -> _KMCurve:
    """Fit a marginal Kaplan-Meier estimate of the *survival-time* distribution.

    Uses the event indicator directly (``event_observed = event``), i.e. the R
    ``survfit(Surv(time, status) ~ 1)`` used for de-censoring. The resulting curve
    estimates ``P(T > t)`` and is used to impute event times for censored calibration
    points conditioned on ``T > C``.
    """
    from lifelines import KaplanMeierFitter

    kmf = KaplanMeierFitter()
    kmf.fit(df["time"].to_numpy(dtype=float),
            event_observed=df["event"].to_numpy(dtype=float))
    sf = kmf.survival_function_
    return _KMCurve(np.asarray(sf.index, dtype=float), sf.iloc[:, 0].to_numpy(dtype=float))


def sample_km_conditional(km: _KMCurve, c: float, rng: np.random.Generator, n: int = 1) -> np.ndarray:
    """Sample ``n`` times from the KM distribution conditioned on ``T > c`` (R ``sample_km_conditional``)."""
    mask = km.time > c
    t_cond = km.time[mask]
    s_cond = km.surv[mask]
    if t_cond.size == 0:
        return np.full(n, float(km.time.max()))

    s_cond = s_cond / s_cond.max()  # renormalize so survival is 1 just after c
    u = rng.uniform(size=n)
    out = np.empty(n, dtype=float)
    for k, x in enumerate(u):
        idx = np.nonzero(s_cond >= x)[0]
        out[k] = t_cond[idx.max()] if idx.size > 0 else float(t_cond.max())
    return out


def impute_km_decensored(
    cal_df: pd.DataFrame,
    km: _KMCurve,
    R: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """De-censor the calibration set by imputing event times for censored rows.

    For each censored calibration observation, sample an imputed event time from the
    marginal KM curve conditioned on ``T > C_observed`` (Qi et al.). Returns ``R`` stacked
    imputations. Only the ``time`` column is modified; ``predict_CQR`` uses ``time`` alone.
    """
    if rng is None:
        rng = np.random.default_rng()

    y = cal_df["time"].to_numpy(dtype=float)
    is_event = cal_df["event"].to_numpy(dtype=float).astype(bool)
    idx_censored = np.nonzero(~is_event)[0]

    frames = []
    for _ in range(R):
        y_new = y.copy()
        for i in idx_censored:
            y_new[i] = sample_km_conditional(km, y[i], rng, n=1)[0]
        imputed = cal_df.copy()
        imputed["time"] = y_new
        frames.append(imputed)

    return pd.concat(frames, ignore_index=True)


def predict_decensoring(
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    wrapper: CoxSurvivalWrapper,
    km: _KMCurve,
    alpha: float,
    R: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """KM-decensored CQR (R ``predict_decensoring``): decensor calibration, then run CQR."""
    cal_decensored = impute_km_decensored(cal_df, km, R=R, rng=rng)
    return predict_CQR(cal_decensored, test_df, wrapper, alpha)


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def evaluate_bounds(
    observed_time: np.ndarray,
    lower_bound: np.ndarray,
    status: Optional[np.ndarray] = None,
) -> dict:
    """Evaluate lower predictive bounds (MIMIC-adapted R ``evaluate_bounds``).

    ``coverage_lower`` uses the observed time Y = min(T, C) and is the quantity that must be
    >= 1 - alpha. For uncensored patients Y = T exactly, so ``coverage_events`` is a direct
    (partial) estimate of true-event-time coverage. True event times for censored patients
    are unknown in MIMIC, so oracle coverage is not reported.
    """
    observed_time = np.asarray(observed_time, dtype=float)
    lower_bound = np.asarray(lower_bound, dtype=float)

    result = {
        "coverage_lower": float(np.mean(lower_bound <= observed_time)),
        "mean_lower_bound": float(np.mean(lower_bound)),
        "median_lower_bound": float(np.median(lower_bound)),
    }

    if status is not None:
        status = np.asarray(status).astype(bool)
        if status.any():
            lb_e = lower_bound[status]
            ot_e = observed_time[status]
            result["coverage_events"] = float(np.mean(lb_e <= ot_e))
            # R censoring-adjusted upper-bound-on-coverage (utils_conformal.R:21).
            result["coverage_upper"] = float(
                1.0 - np.mean(lb_e > ot_e) * np.mean(status)
            )
        else:
            result["coverage_events"] = np.nan
            result["coverage_upper"] = np.nan

    return result


# --------------------------------------------------------------------------------------
# Convenience driver for the SigBERT pipeline outputs
# --------------------------------------------------------------------------------------
def run_cqr_conformal(
    df_survival_test_list: Sequence[pd.DataFrame],
    cph,
    cal_group_idx: Sequence[int],
    alpha: float = 0.1,
    method: str = "cqr",
    R: int = 1,
    random_state: int = 0,
):
    """Run CQR conformal prediction over the pipeline's held-out test groups.

    The Cox model was trained on a separate training split, so every ``df_survival`` in
    ``df_survival_test_list`` carries out-of-sample risk scores -- valid for conformal
    calibration. The groups in ``cal_group_idx`` are pooled as the calibration set; the
    remaining groups form the conformal test set.

    Parameters
    ----------
    df_survival_test_list : list of DataFrame
        Each has columns ``event``, ``time``, ``risk_score`` (and optionally ``ID``).
    cph : lifelines.CoxPHFitter
        Trained Cox model (only ``baseline_survival_`` is used).
    cal_group_idx : sequence of int
        0-based indices of the groups to pool as the calibration set.
    alpha : float, default 0.1
        Miscoverage level (90% coverage by default).
    method : {"cqr", "decensored"}
        Plain CQR, or KM-decensored CQR.
    R : int, default 1
        Number of KM imputations (``method="decensored"`` only).
    random_state : int
        Seed for the imputation RNG.

    Returns
    -------
    results_df : DataFrame
        One row per test group with coverage metrics and mean/median LPB (days).
    lpb_by_group : dict[int, numpy.ndarray]
        Per-group lower predictive bounds, keyed by the group's 0-based index.
    """
    if method not in ("cqr", "decensored"):
        raise ValueError(f"Unknown method: {method!r} (expected 'cqr' or 'decensored').")

    rng = np.random.default_rng(random_state)
    wrapper = CoxSurvivalWrapper(cph.baseline_survival_.squeeze())

    cal_set = set(int(i) for i in cal_group_idx)
    cal_df = pd.concat([df_survival_test_list[i] for i in sorted(cal_set)], ignore_index=True)
    test_idx = [i for i in range(len(df_survival_test_list)) if i not in cal_set]

    km = fit_survival_km(cal_df) if method == "decensored" else None

    rows = []
    lpb_by_group = {}
    for i in test_idx:
        test_df = df_survival_test_list[i]
        if method == "cqr":
            lpb = predict_CQR(cal_df, test_df, wrapper, alpha)
        else:
            lpb = predict_decensoring(cal_df, test_df, wrapper, km, alpha, R=R, rng=rng)

        metrics = evaluate_bounds(
            test_df["time"].to_numpy(dtype=float),
            lpb,
            status=test_df["event"].to_numpy(dtype=float),
        )
        metrics = {"group": i + 1, "n_patients": len(test_df), **metrics}
        rows.append(metrics)
        lpb_by_group[i] = lpb

    results_df = pd.DataFrame(rows)
    return results_df, lpb_by_group



# --- Aggregate across evaluation groups (mean +/- 95% jackknife CI) -------------
def summarize_conformal(results_df, label):
    rows = []
    for col in ["coverage_lower", "coverage_events", "mean_lower_bound"]:
        vals = results_df[col].to_numpy(dtype=float)
        lo, hi = jackknife_confidence_interval(vals)
        rows.append({
            "Method": label, "Metric": col,
            "Mean": np.mean(vals), "Std": np.std(vals, ddof=1),
            "95% CI Lower": lo, "95% CI Upper": hi,
        })
    return pd.DataFrame(rows)






def _jackknife_mean_ci(values, confidence_level=0.95):
    """Return the mean and a jackknife confidence interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)
    mean = float(np.mean(values))

    if n < 2:
        return mean, np.nan, np.nan

    total = np.sum(values)
    loo_means = (total - values) / (n - 1)
    loo_mean = np.mean(loo_means)

    se = np.sqrt(
        (n - 1) / n
        * np.sum((loo_means - loo_mean) ** 2)
    )

    z = NormalDist().inv_cdf(
        0.5 + confidence_level / 2
    )

    return mean, mean - z * se, mean + z * se


def plot_conformal_results(
    results_cqr,
    results_decensored,
    lpb_cqr,
    lpb_decensored,
    alpha=0.1,
    bins=40,
    confidence_level=0.95,
    figsize=(13, 4.5),
    colors=("#3DCEB7", "#8042EF"),
    method_labels=("CQR (naive)", "CQR (de-censored)"),
    show=True,
):
    """Plot LPB distributions and empirical coverage with confidence intervals.

    Parameters
    ----------
    results_cqr, results_decensored : pandas.DataFrame
        Outputs of ``run_cqr_conformal`` for the two methods.
    lpb_cqr, lpb_decensored : dict
        Lower predictive bounds returned by ``run_cqr_conformal``.
    alpha : float
        Miscoverage level.
    confidence_level : float
        Confidence level for jackknife intervals across evaluation groups.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ------------------------------------------------------------------
    # Collect patient-level lower predictive bounds
    # ------------------------------------------------------------------
    lpb_cqr_all = np.concatenate([
        np.asarray(lpb_cqr[i], dtype=float)
        for i in sorted(lpb_cqr)
    ])

    lpb_dec_all = np.concatenate([
        np.asarray(lpb_decensored[i], dtype=float)
        for i in sorted(lpb_decensored)
    ])

    bounds_df = pd.DataFrame({
        "Lower bound": np.concatenate([
            lpb_cqr_all,
            lpb_dec_all,
        ]),
        "Method": (
            [method_labels[0]] * len(lpb_cqr_all)
            + [method_labels[1]] * len(lpb_dec_all)
        ),
    })

    palette = {
        method_labels[0]: colors[0],
        method_labels[1]: colors[1],
    }

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
    )

    # ------------------------------------------------------------------
    # Left panel: distribution of lower predictive bounds
    # ------------------------------------------------------------------
    sns.histplot(
        data=bounds_df,
        x="Lower bound",
        hue="Method",
        bins=bins,
        multiple="layer",
        alpha=0.55,
        palette=palette,
        ax=axes[0],
    )

    axes[0].set_xlabel(
        r"Lower bound on survival time $L(x)$ (days)"
    )
    axes[0].set_ylabel("Number of test patients")
    axes[0].set_title(
        f"{int((1 - alpha) * 100)}% lower predictive bounds"
    )

    # ------------------------------------------------------------------
    # Right panel: empirical coverage
    # ------------------------------------------------------------------
    metrics = {
        "coverage_lower": "Coverage on\nobserved time $Y$",
        "coverage_events": "Coverage on\nevents ($Y=T$)",
    }

    coverage_rows = []

    for method, df in [
        (method_labels[0], results_cqr),
        (method_labels[1], results_decensored),
    ]:
        for metric, label in metrics.items():
            for value in df[metric].to_numpy(dtype=float):
                coverage_rows.append({
                    "Method": method,
                    "Metric": label,
                    "Coverage": value,
                })

    coverage_df = pd.DataFrame(coverage_rows)

    # Draw bars without Seaborn's default bootstrap CI.
    sns.barplot(
        data=coverage_df,
        x="Metric",
        y="Coverage",
        hue="Method",
        estimator=np.mean,
        errorbar=None,
        palette=palette,
        ax=axes[1],
    )

    # ------------------------------------------------------------------
    # Add jackknife confidence intervals manually
    # ------------------------------------------------------------------
    metric_names = list(metrics.values())
    n_methods = len(method_labels)

    total_width = 0.8
    bar_width = total_width / n_methods

    for metric_idx, metric_label in enumerate(metric_names):
        for method_idx, method in enumerate(method_labels):

            values = coverage_df.loc[
                (coverage_df["Metric"] == metric_label)
                & (coverage_df["Method"] == method),
                "Coverage",
            ].to_numpy(dtype=float)

            mean, ci_lower, ci_upper = _jackknife_mean_ci(values,confidence_level=confidence_level)

            x_position = (
                metric_idx
                - total_width / 2
                + bar_width / 2
                + method_idx * bar_width
            )

            yerr = np.array([
                [mean - ci_lower],
                [ci_upper - mean],
            ])

            axes[1].errorbar(
                x_position,
                mean,
                yerr=yerr,
                fmt="none",
                ecolor="black",
                elinewidth=1.2,
                capsize=4,
                capthick=1.2,
            )

    nominal_coverage = 1.0 - alpha

    axes[1].axhline(
        nominal_coverage,
        linestyle="--",
        color="black",
        linewidth=1.2,
        label=f"Nominal {nominal_coverage:.2f}",
    )

    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Empirical coverage")
    axes[1].set_title("Coverage vs nominal level")

    # Rebuild legend to include the nominal coverage line.
    handles, labels = axes[1].get_legend_handles_labels()

    unique = dict(zip(labels, handles))
    axes[1].legend(
        unique.values(),
        unique.keys(),
        frameon=False,
    )
    sns.despine(fig=fig)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes