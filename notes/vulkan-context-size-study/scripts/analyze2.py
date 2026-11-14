#!/usr/bin/env python3
"""Aggregate the context sweeps and draw the charts used by the report.

Error bars are 95% intervals:
  - worst NMSE per run is averaged on a log10 scale and the interval is a
    Student t interval over the repetitions. A zero width is a real result: the
    repetitions produced bit identical numbers.
  - the pass rate interval is a Wilson score interval over the case verdicts.

    ./analyze2.py pre-short.csv post-short.csv pre-scaled.csv post-scaled.csv
"""

import csv
import math
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
       8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179}

TOPO = {
    "die1": dict(color="#1b7f4b", marker="o", label="die 1 only"),
    "die2": dict(color="#1f5fa8", marker="s", label="die 2 only"),
    "both": dict(color="#b03030", marker="^", label="both dies (layer split)"),
}

STATE = {
    "pre":  dict(color="#b03030", marker="o", linestyle="--", label="before the fix"),
    "post": dict(color="#1b7f4b", marker="o", linestyle="-",  label="after the fix"),
}


def t_crit(n):
    return T95.get(n, 1.96) if n > 1 else 0.0


def mean_ci(vals):
    v = np.asarray([x for x in vals if not math.isnan(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), 0.0
    return float(v.mean()), t_crit(v.size)*float(v.std(ddof=1))/math.sqrt(v.size)


def wilson(k, n, z=1.96):
    """Point estimate and the half width of its Wilson score interval."""
    if n == 0:
        return float("nan"), float("nan")
    p = k/n
    half = z*math.sqrt(p*(1 - p)/n + z*z/(4*n*n))/(1 + z*z/n)
    return p, half


def load(paths):
    rows = []
    for p in paths:
        mode = "short" if "short" in p else "scaled"
        state = "pre" if p.split("/")[-1].startswith("pre") else "post"
        with open(p) as f:
            for r in csv.DictReader(f):
                r["ctx"] = int(r["ctx"])
                for k in ("cases_pass", "cases_fail", "cases_skip", "prompt_tokens"):
                    r[k] = int(r[k])
                r["seconds"] = float(r["seconds"])
                r["max_nmse"] = float(r["max_nmse"]) if r["max_nmse"] not in ("", "nan") else float("nan")
                r["mode"] = r.get("prompt_mode") or mode
                if r["mode"] == "builtin":
                    r["mode"] = "short"
                r["state"] = state
                rows.append(r)
    return rows


def aggregate(rows):
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g[(r["mode"], r["state"], r["topology"])][r["ctx"]].append(r)

    out = {}
    for key, by_ctx in g.items():
        out[key] = {}
        for ctx, rs in sorted(by_ctx.items()):
            logs = [math.log10(r["max_nmse"]) if r["max_nmse"] > 0 else float("nan") for r in rs]
            mlog, cilog = mean_ci(logs)
            npass = sum(r["cases_pass"] for r in rs)
            nfail = sum(r["cases_fail"] for r in rs)
            p, ph = wilson(npass, npass + nfail)
            sec, sech = mean_ci([r["seconds"] for r in rs])
            out[key][ctx] = dict(
                n_reps=len(rs),
                nmse=10**mlog if not math.isnan(mlog) else float("nan"),
                nmse_lo=10**(mlog - cilog) if not math.isnan(mlog) else float("nan"),
                nmse_hi=10**(mlog + cilog) if not math.isnan(mlog) else float("nan"),
                pass_rate=p, pass_rate_ci=ph,
                cases_pass=npass, cases_fail=nfail,
                runs_failed=sum(1 for r in rs if r["cases_fail"] > 0),
                seconds=sec, seconds_ci=sech,
            )
    return out


def err_nmse(cells, ctxs):
    y = [cells[c]["nmse"] for c in ctxs]
    lo = [cells[c]["nmse"] - cells[c]["nmse_lo"] for c in ctxs]
    hi = [cells[c]["nmse_hi"] - cells[c]["nmse"] for c in ctxs]
    return y, [lo, hi]


def style_axes(ax, ylabel, title, xlabel="context size (tokens)"):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({128, 256, 512, 1024, 2048, 4096, 8192}))
    ax.set_xticklabels(["128", "256", "512", "1k", "2k", "4k", "8k"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)


def fig_both_mode(agg, mode, fname, title):
    """Before and after, both dies, worst NMSE and pass rate."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    for state, st in STATE.items():
        cells = agg.get((mode, state, "both"))
        if not cells:
            continue
        ctxs = sorted(cells)
        y, yerr = err_nmse(cells, ctxs)
        axes[0].errorbar(ctxs, y, yerr=yerr, capsize=3, linewidth=1.6, markersize=5,
                         color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                         label=st["label"])
        axes[1].errorbar(ctxs, [100*cells[c]["pass_rate"] for c in ctxs],
                         yerr=[100*cells[c]["pass_rate_ci"] for c in ctxs],
                         capsize=3, linewidth=1.6, markersize=5,
                         color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                         label=st["label"])

    axes[0].set_yscale("log")
    axes[0].axhline(1e-5, color="grey", linestyle=":", linewidth=1)
    axes[0].text(130, 2e-5, "NMSE threshold 1e-5", fontsize=7, color="grey")
    style_axes(axes[0], "worst NMSE of the run", f"worst NMSE, {title}")
    style_axes(axes[1], "case verdicts passed (%)", f"case pass rate, {title}")
    axes[1].set_ylim(-5, 105)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def fig_topologies(agg, mode, state, fname, title, metric="nmse"):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for topo, ts in TOPO.items():
        cells = agg.get((mode, state, topo))
        if not cells:
            continue
        ctxs = sorted(cells)
        if metric == "nmse":
            y, yerr = err_nmse(cells, ctxs)
            ax.errorbar(ctxs, y, yerr=yerr, capsize=3, linewidth=1.6, markersize=5, **ts)
        elif metric == "pass":
            ax.errorbar(ctxs, [100*cells[c]["pass_rate"] for c in ctxs],
                        yerr=[100*cells[c]["pass_rate_ci"] for c in ctxs],
                        capsize=3, linewidth=1.6, markersize=5, **ts)
        else:
            ax.errorbar(ctxs, [cells[c]["seconds"] for c in ctxs],
                        yerr=[cells[c]["seconds_ci"] for c in ctxs],
                        capsize=3, linewidth=1.6, markersize=5, **ts)
    if metric == "nmse":
        ax.set_yscale("log")
        ax.axhline(1e-5, color="grey", linestyle=":", linewidth=1)
        style_axes(ax, "worst NMSE of the run", title)
    elif metric == "pass":
        ax.set_ylim(-5, 105)
        style_axes(ax, "case verdicts passed (%)", title)
    else:
        style_axes(ax, "wall clock of one sweep (s)", title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def markdown_table(agg):
    lines = ["| state | prompt | topology | ctx | pass rate (%) | worst NMSE (95% CI) | runs with a failure | wall clock (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for mode in ("short", "scaled"):
        for state in ("pre", "post"):
            for topo in ("die1", "die2", "both"):
                cells = agg.get((mode, state, topo), {})
                for ctx in sorted(cells):
                    c = cells[ctx]
                    lines.append(
                        f"| {state} | {mode} | {topo} | {ctx} | "
                        f"{100*c['pass_rate']:.1f} +/- {100*c['pass_rate_ci']:.1f} | "
                        f"{c['nmse']:.2e} ({c['nmse_lo']:.2e} .. {c['nmse_hi']:.2e}) | "
                        f"{c['runs_failed']}/{c['n_reps']} | "
                        f"{c['seconds']:.2f} +/- {c['seconds_ci']:.2f} |")
    return "\n".join(lines)


def main():
    paths = sys.argv[1:]
    rows = load(paths)
    agg = aggregate(rows)

    plt.rcParams.update({"font.size": 9, "figure.dpi": 140,
                         "axes.titlesize": 10, "figure.facecolor": "white"})

    fig_both_mode(agg, "short", "chart-short-both.png", "built in 100 token prompt")
    fig_both_mode(agg, "scaled", "chart-scaled-both.png", "prompt scaled to half the context")

    fig_topologies(agg, "short", "pre", "chart-short-pre-topologies.png",
                   "before the fix, all topologies (worst NMSE)")
    fig_topologies(agg, "short", "post", "chart-short-post-topologies.png",
                   "after the fix, all topologies (worst NMSE)")
    fig_topologies(agg, "scaled", "post", "chart-scaled-post-topologies.png",
                   "after the fix, scaled prompts (worst NMSE)")
    fig_topologies(agg, "scaled", "post", "chart-time.png",
                   "after the fix (wall clock)", metric="time")

    with open("table.md", "w") as f:
        f.write(markdown_table(agg) + "\n")

    print(markdown_table(agg))
    print("\nwrote: chart-short-both.png chart-scaled-both.png "
          "chart-short-pre-topologies.png chart-short-post-topologies.png "
          "chart-scaled-post-topologies.png chart-time.png table.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
