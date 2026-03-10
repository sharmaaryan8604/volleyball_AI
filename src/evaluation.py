"""
evaluation.py
-------------
Comprehensive evaluation metrics for the volleyball zone prediction model.

Functions:
    full_evaluation_report   — prints all metrics in one call
    per_zone_accuracy        — top-1 accuracy broken down by true zone
    confusion_analysis       — most common prediction mistakes
    calibration_report       — how well predicted probs match actual freq
    zone_coverage_report     — top-k zone coverage statistics
    mrr_by_hitter_zone       — MRR broken down by hitter location
    expected_calibration_error — ECE scalar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import top_k_accuracy_score, log_loss


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _mrr(y_true, probs):
    ranks = []
    for i in range(len(y_true)):
        order = np.argsort(probs[i])[::-1]
        match = np.where(order == int(y_true.iloc[i]))[0]
        ranks.append(1 / (match[0] + 1) if len(match) > 0 else 0)
    return np.mean(ranks), ranks


def _hits_at_k(y_true, probs, k):
    n = len(y_true)
    hits = sum(
        int(y_true.iloc[i]) in np.argsort(probs[i])[::-1][:k]
        for i in range(n)
    )
    return hits / n


# ─────────────────────────────────────────────────────────────
# 1. Full report (single call)
# ─────────────────────────────────────────────────────────────

def full_evaluation_report(y_test, probs, y_train=None,
                            X_test=None, label="Model"):
    """
    Print a complete evaluation report and return all metrics as a dict.

    Args:
        y_test   : pd.Series of true labels (0-indexed)
        probs    : np.ndarray shape (n, n_classes) — predicted probabilities
        y_train  : optional — for baseline comparisons
        X_test   : optional — for per-hitter-zone breakdown
        label    : string label for the model
    """
    n_classes = probs.shape[1]

    top1 = top_k_accuracy_score(y_test, probs, k=1)
    top3 = top_k_accuracy_score(y_test, probs, k=3)
    top5 = top_k_accuracy_score(y_test, probs, k=5)
    mrr, _ = _mrr(y_test, probs)

    ll = log_loss(y_test, probs)
    ece = expected_calibration_error(y_test, probs)

    # Baselines
    random_top1   = 1 / n_classes
    if y_train is not None:
        counts = np.bincount(y_train, minlength=n_classes)
        maj_class = counts.argmax()
        maj_top1  = (y_test == maj_class).mean()
    else:
        maj_top1 = None

    print(f"\n{'='*55}")
    print(f"  EVALUATION REPORT — {label}")
    print(f"{'='*55}")
    print(f"  Samples evaluated : {len(y_test):,}")
    print(f"  Classes           : {n_classes}")
    print(f"{'─'*55}")
    print(f"  {'Metric':<28} {'Value':>10}  {'vs Random':>10}")
    print(f"{'─'*55}")

    metrics = [
        ("Top-1 Accuracy",   top1,  random_top1),
        ("Top-3 Accuracy",   top3,  min(3/n_classes, 1.0)),
        ("Top-5 Accuracy",   top5,  min(5/n_classes, 1.0)),
        ("MRR",              mrr,   2/(n_classes+1)),
        ("Log Loss",         ll,    np.log(n_classes)),
        ("ECE (calibration)",ece,   None),
    ]

    for name, val, baseline in metrics:
        if baseline is not None:
            lift = f"+{(val-baseline)*100:+.1f}pp" if name not in ("Log Loss",) \
                   else f"{val-baseline:+.3f}"
        else:
            lift = "—"
        print(f"  {name:<28} {val:>10.4f}  {lift:>10}")

    if maj_top1 is not None:
        print(f"{'─'*55}")
        print(f"  {'Majority class baseline':<28} {maj_top1:>10.4f}")
        print(f"  {'Lift over majority':<28} {(top1-maj_top1)*100:>+9.2f}pp")

    print(f"{'='*55}\n")

    return {
        "top1": top1, "top3": top3, "top5": top5,
        "mrr": mrr, "log_loss": ll, "ece": ece,
    }


# ─────────────────────────────────────────────────────────────
# 2. Per-zone accuracy
# ─────────────────────────────────────────────────────────────

def per_zone_accuracy(y_test, probs, k=1):
    """
    Top-k accuracy broken down by true landing zone.

    Returns:
        pd.DataFrame — zone, n_samples, top_k_accuracy, hardest zones
    """
    results = []
    zones   = sorted(y_test.unique())

    for zone in zones:
        mask = y_test == zone
        if mask.sum() == 0:
            continue
        y_sub = y_test[mask]
        p_sub = probs[mask]
        acc   = _hits_at_k(y_sub, p_sub, k)
        results.append({
            "zone":        zone + 1,   # back to 1-indexed for display
            "n_samples":   int(mask.sum()),
            f"top{k}_acc": round(acc, 4),
        })

    df = pd.DataFrame(results).set_index("zone")
    df = df.sort_values(f"top{k}_acc", ascending=True)
    return df


# ─────────────────────────────────────────────────────────────
# 3. Confusion analysis
# ─────────────────────────────────────────────────────────────

def confusion_analysis(y_test, probs, top_n=10):
    """
    Show the most common prediction errors.

    Returns:
        pd.DataFrame — true_zone, predicted_zone, error_count, error_rate
    """
    pred  = np.argmax(probs, axis=1)
    wrong = [(int(y_test.iloc[i])+1, pred[i]+1)
             for i in range(len(y_test)) if pred[i] != int(y_test.iloc[i])]

    from collections import Counter
    counts = Counter(wrong)
    total  = len(y_test)

    rows = []
    for (true_z, pred_z), cnt in counts.most_common(top_n):
        rows.append({
            "true_zone":  true_z,
            "pred_zone":  pred_z,
            "errors":     cnt,
            "error_rate%": round(cnt / total * 100, 2),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 4. Calibration report
# ─────────────────────────────────────────────────────────────

def calibration_report(y_test, probs, n_bins=10):
    """
    Reliability diagram data — are predicted probabilities accurate?

    Returns:
        pd.DataFrame — bin_midpoint, mean_predicted_prob, actual_freq, gap
    """
    y_onehot = np.zeros_like(probs)
    for i, label in enumerate(y_test):
        y_onehot[i, int(label)] = 1

    flat_pred   = probs.flatten()
    flat_actual = y_onehot.flatten()

    bins = np.linspace(0, 1, n_bins + 1)
    rows = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (flat_pred >= lo) & (flat_pred < hi)
        if mask.sum() == 0:
            continue
        mean_pred = flat_pred[mask].mean()
        actual    = flat_actual[mask].mean()
        rows.append({
            "bin":            f"{lo:.1f}–{hi:.1f}",
            "mean_pred_prob": round(mean_pred, 4),
            "actual_freq":    round(actual, 4),
            "gap":            round(mean_pred - actual, 4),
            "n_samples":      int(mask.sum()),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 5. Zone coverage report
# ─────────────────────────────────────────────────────────────

def zone_coverage_report(y_test, probs):
    """
    For each k from 1 to 10, show % of test samples where
    the true zone appears in the top-k predictions.

    Returns:
        pd.DataFrame — k, coverage%, avg_rank
    """
    rows = []
    for k in range(1, 11):
        cov = _hits_at_k(y_test, probs, k)
        rows.append({"k": k, "coverage%": round(cov * 100, 2)})
    return pd.DataFrame(rows).set_index("k")


# ─────────────────────────────────────────────────────────────
# 6. MRR broken down by hitter zone
# ─────────────────────────────────────────────────────────────

def mrr_by_hitter_zone(y_test, probs, X_test):
    """
    MRR split by hitter_location — shows which hitter positions
    the model predicts best/worst.

    Args:
        y_test  : true labels
        probs   : predicted probabilities
        X_test  : feature DataFrame (must contain 'hitter_location')

    Returns:
        pd.DataFrame — hitter_zone, n_attacks, mrr, top1_acc
    """
    if "hitter_location" not in X_test.columns:
        raise ValueError("X_test must contain 'hitter_location'")

    rows   = []
    zones  = sorted(X_test["hitter_location"].dropna().unique())

    for zone in zones:
        mask = (X_test["hitter_location"].fillna(-999) == zone).values.astype(bool)
        if mask.sum() < 5:
            continue
        idx   = np.where(mask)[0]
        y_sub = y_test.iloc[idx]
        p_sub = probs[idx]
        mrr_val, _ = _mrr(y_sub, p_sub)
        top1        = _hits_at_k(y_sub, p_sub, 1)
        rows.append({
            "hitter_zone": int(zone),
            "n_attacks":   int(mask.sum()),
            "mrr":         round(mrr_val, 4),
            "top1_acc":    round(top1, 4),
        })

    return pd.DataFrame(rows).set_index("hitter_zone").sort_values("mrr", ascending=False)


# ─────────────────────────────────────────────────────────────
# 7. Expected Calibration Error
# ─────────────────────────────────────────────────────────────

def expected_calibration_error(y_test, probs, n_bins=15):
    """
    ECE — scalar measure of probability calibration.
    Lower is better. Well-calibrated model: ECE < 0.05.
    """
    y_onehot  = np.zeros_like(probs)
    for i, label in enumerate(y_test):
        y_onehot[i, int(label)] = 1

    flat_pred   = probs.flatten()
    flat_actual = y_onehot.flatten()

    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(flat_pred)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (flat_pred >= lo) & (flat_pred < hi)
        if mask.sum() == 0:
            continue
        conf = flat_pred[mask].mean()
        acc  = flat_actual[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)

    return round(ece, 5)


# ─────────────────────────────────────────────────────────────
# 8. Visual dashboard (all plots in one figure)
# ─────────────────────────────────────────────────────────────

def plot_evaluation_dashboard(y_test, probs, X_test=None, title="Model Evaluation"):
    """
    One-call visual evaluation dashboard with 6 subplots:
      1. Top-k coverage curve
      2. Per-zone Top-1 accuracy (bar)
      3. Calibration reliability diagram
      4. MRR by hitter zone (if X_test provided)
      5. Confidence distribution (histogram)
      6. Top-10 error pairs (heatmap)
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Top-k coverage ────────────────────────────────────
    ax1  = fig.add_subplot(gs[0, 0])
    cov  = zone_coverage_report(y_test, probs)
    ax1.plot(cov.index, cov["coverage%"], marker="o", color="#2980b9", lw=2)
    ax1.fill_between(cov.index, cov["coverage%"], alpha=0.15, color="#2980b9")
    ax1.set_xlabel("k"); ax1.set_ylabel("Coverage %")
    ax1.set_title("Top-k Zone Coverage")
    ax1.set_xticks(range(1, 11))
    ax1.grid(True, alpha=0.3)

    # ── 2. Per-zone accuracy ──────────────────────────────────
    ax2  = fig.add_subplot(gs[0, 1])
    pza  = per_zone_accuracy(y_test, probs, k=1).sort_index()
    colors = ["#e74c3c" if v < 0.4 else "#f39c12" if v < 0.6
              else "#27ae60" for v in pza["top1_acc"]]
    ax2.bar(pza.index.astype(str), pza["top1_acc"] * 100, color=colors)
    ax2.set_xlabel("Landing Zone"); ax2.set_ylabel("Top-1 Accuracy %")
    ax2.set_title("Per-Zone Top-1 Accuracy")
    ax2.tick_params(axis="x", labelsize=7)
    ax2.axhline(y_test.shape[0] and top_k_accuracy_score(y_test, probs, k=1)*100,
                color="black", lw=1, linestyle="--", label="Overall")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    # ── 3. Calibration diagram ────────────────────────────────
    ax3  = fig.add_subplot(gs[0, 2])
    cal  = calibration_report(y_test, probs)
    ax3.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax3.scatter(cal["mean_pred_prob"], cal["actual_freq"],
                c=cal["gap"].abs(), cmap="RdYlGn_r", s=80, zorder=5)
    ax3.set_xlabel("Mean Predicted Probability")
    ax3.set_ylabel("Actual Frequency")
    ax3.set_title(f"Calibration Diagram  (ECE={expected_calibration_error(y_test,probs):.4f})")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # ── 4. MRR by hitter zone ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if X_test is not None and "hitter_location" in X_test.columns:
        mhz = mrr_by_hitter_zone(y_test, probs, X_test)
        ax4.barh(mhz.index.astype(str), mhz["mrr"], color="#8e44ad")
        ax4.set_xlabel("MRR"); ax4.set_ylabel("Hitter Zone")
        ax4.set_title("MRR by Hitter Zone")
        ax4.axvline(mhz["mrr"].mean(), color="red", lw=1.5,
                    linestyle="--", label=f"Mean {mhz['mrr'].mean():.3f}")
        ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3, axis="x")
    else:
        ax4.text(0.5, 0.5, "X_test not provided\n(pass X_test= to enable)",
                 ha="center", va="center", transform=ax4.transAxes, fontsize=10)
        ax4.set_title("MRR by Hitter Zone")

    # ── 5. Confidence distribution ────────────────────────────
    ax5    = fig.add_subplot(gs[1, 1])
    max_p  = probs.max(axis=1)
    correct = np.array([int(y_test.iloc[i]) == np.argmax(probs[i])
                        for i in range(len(y_test))])
    ax5.hist(max_p[correct],  bins=30, alpha=0.6, color="#27ae60", label="Correct")
    ax5.hist(max_p[~correct], bins=30, alpha=0.6, color="#e74c3c", label="Wrong")
    ax5.set_xlabel("Max Predicted Probability")
    ax5.set_ylabel("Count")
    ax5.set_title("Confidence Distribution")
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    # ── 6. Top confusion pairs heatmap ───────────────────────
    ax6   = fig.add_subplot(gs[1, 2])
    conf  = confusion_analysis(y_test, probs, top_n=100)
    n_zones = 15
    matrix  = np.zeros((n_zones, n_zones))
    for _, row in conf.iterrows():
        t = int(row["true_zone"]) - 1
        p = int(row["pred_zone"]) - 1
        if 0 <= t < n_zones and 0 <= p < n_zones:
            matrix[t, p] = row["errors"]
    np.fill_diagonal(matrix, 0)
    sns.heatmap(matrix, ax=ax6, cmap="YlOrRd",
                xticklabels=range(1, n_zones+1),
                yticklabels=range(1, n_zones+1),
                linewidths=0.3, cbar_kws={"shrink": 0.7})
    ax6.set_xlabel("Predicted Zone"); ax6.set_ylabel("True Zone")
    ax6.set_title("Confusion Heatmap (errors only)")
    ax6.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig("evaluation_dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Dashboard saved to evaluation_dashboard.png")