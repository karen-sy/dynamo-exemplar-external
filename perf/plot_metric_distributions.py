#!/usr/bin/env python3
"""
Plot metric distributions (TTFT, ITL, E2E) from aiperf profile_export.jsonl traces.

Each trace is a row; each metric is a column. X-axis uses log scale automatically
when the value range spans more than 20x. Each subplot has its own independent
x-range.

Usage:
    python plot_metric_distributions.py profile_export1.jsonl profile_export2.jsonl \
        --labels "Agg" "Disagg" \
        --title "My Experiment" \
        -o out.png
"""
import argparse
import gzip
import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

MetricExtractor = Callable[[Dict], Optional[float]]


def _extract_scalar(rec: Dict, key: str) -> Optional[float]:
    v = (rec.get("metrics") or {}).get(key)
    if isinstance(v, dict):
        v = v.get("value")
    return float(v) if isinstance(v, (int, float)) else None


def extract_ttft_ms(rec: Dict) -> Optional[float]:
    return _extract_scalar(rec, "time_to_first_token")


def extract_itl_ms(rec: Dict) -> Optional[float]:
    v = _extract_scalar(rec, "inter_token_latency")
    if v is not None:
        return v
    icl = (rec.get("metrics") or {}).get("inter_chunk_latency")
    if isinstance(icl, dict) and isinstance(icl.get("value"), list) and icl["value"]:
        vals = [float(x) for x in icl["value"] if isinstance(x, (int, float))]
        return float(np.mean(vals)) if vals else None
    return None


def extract_e2e_ms(rec: Dict) -> Optional[float]:
    return _extract_scalar(rec, "request_latency")


METRIC_SPECS: Dict[str, Tuple[str, MetricExtractor]] = {
    "ttft": ("Time to First Token (ms)", extract_ttft_ms),
    "itl":  ("Inter-token Latency (ms)", extract_itl_ms),
    "e2e":  ("End-to-End Latency (ms)",  extract_e2e_ms),
}

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def iter_jsonl(path: str) -> Iterable[Dict]:
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def derive_label(path: str) -> str:
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if not parent:
        parent = os.path.splitext(os.path.basename(path))[0]
    _, tail = parent.split("_", 1) if "_" in parent else (parent, "")
    return tail.strip() or parent


def discover_traces(dir_roots: List[str], dir_regex: Optional[str], trace_regex: str) -> List[str]:
    traces: List[str] = []
    dir_pattern = re.compile(dir_regex) if dir_regex else None
    trace_pattern = re.compile(trace_regex)
    for root in dir_roots:
        root_path = Path(root).expanduser().resolve()
        if not root_path.is_dir():
            continue
        search_dirs: List[Path] = []
        if not dir_pattern or dir_pattern.search(root_path.name):
            search_dirs.append(root_path)
        for child in sorted(root_path.iterdir()):
            if child.is_dir() and (not dir_pattern or dir_pattern.search(child.name)):
                search_dirs.append(child)
        for directory in search_dirs:
            for candidate in sorted(directory.rglob("*")):
                if candidate.is_file() and trace_pattern.search(candidate.name):
                    traces.append(str(candidate))
    return traces


def collect_metrics(
    paths: List[str], metric_keys: List[str]
) -> Tuple[List[str], Dict[str, List[np.ndarray]]]:
    labels = [derive_label(p) for p in paths]
    metric_data: Dict[str, List[np.ndarray]] = {METRIC_SPECS[k][0]: [] for k in metric_keys}
    for p in paths:
        per_metric: Dict[str, List[float]] = {k: [] for k in metric_keys}
        for rec in iter_jsonl(p):
            for mk in metric_keys:
                val = METRIC_SPECS[mk][1](rec)
                if val is not None:
                    per_metric[mk].append(val)
        for mk in metric_keys:
            metric_data[METRIC_SPECS[mk][0]].append(np.array(per_metric[mk], dtype=float))
    return labels, metric_data

# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def gaussian_kernel_1d(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return np.array([1.0], dtype=float)
    radius = max(1, int(round(3.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    return k / k.sum()

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_distributions(
    labels: List[str],
    metric_names: List[str],
    metric_data: Dict[str, List[np.ndarray]],
    title: Optional[str],
    output: str,
    bins: int,
    sigma: float,
) -> None:
    nrows, ncols = len(labels), len(metric_names)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(max(6 * ncols, 6), max(3.4 * nrows, 4)),
        squeeze=False,
    )
    kernel = gaussian_kernel_1d(sigma)

    for row_idx, label in enumerate(labels):
        for col_idx, metric_name in enumerate(metric_names):
            ax = axes[row_idx][col_idx]
            values = metric_data[metric_name][row_idx]

            if values.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        fontsize=10, transform=ax.transAxes)
                ax.set_xlim(0, 1)
            else:
                xmin = float(values.min())
                xmax = float(values.max())
                if xmax <= xmin:
                    xmax = xmin + 1.0

                # Auto log-scale when range spans > 20x
                use_log = xmin > 0 and (xmax / xmin) > 20
                if use_log:
                    bin_edges = np.logspace(np.log10(xmin), np.log10(xmax), bins + 1)
                    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
                else:
                    bin_edges = np.linspace(xmin, xmax, bins + 1)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5

                counts, _ = np.histogram(values, bins=bin_edges)
                smoothed = np.convolve(counts.astype(float), kernel, mode="same")
                ax.plot(bin_centers, smoothed, linewidth=2.0)
                if use_log:
                    ax.set_xscale("log")
                ax.set_xlim(xmin, xmax)

                q25, q50, q75, q99 = np.percentile(values, [25, 50, 75, 99])
                mean = float(np.mean(values))
                ax.axvline(mean, color="red",     linestyle=":", linewidth=1.5, alpha=0.9, label=f"Mean: {mean:.1f}")
                ax.axvline(q50,  color="green",   linestyle=":", linewidth=1.5, alpha=0.9, label=f"Median: {q50:.1f}")
                ax.axvline(q25,  color="#7f7f7f", linestyle="--", linewidth=1.2, alpha=0.8)
                ax.axvline(q75,  color="#7f7f7f", linestyle="--", linewidth=1.2, alpha=0.8)
                ax.axvline(q99,  color="#7f7f7f", linestyle="--", linewidth=1.2, alpha=0.8)
                ax.legend(loc="upper right", fontsize=8)

            # Row label at top-left of the first column
            if col_idx == 0:
                ax.set_title(label or f"Trace {row_idx + 1}", loc="left",
                             fontsize=13, fontweight="bold", pad=6)
            # Metric column header elevated above row labels to avoid overlap
            if row_idx == 0:
                ax.set_title(metric_name, loc="center", fontsize=12, y=1.10)

            ax.set_ylabel("Request count", fontsize=10)
            if row_idx == nrows - 1:
                ax.set_xlabel("Time (ms)", fontsize=10)
            ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    if title:
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        plt.tight_layout()

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close()
    print(f"Wrote {output}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot smoothed request-count latency distributions from aiperf traces."
    )
    ap.add_argument("traces", nargs="*", help="profile_export.jsonl(.gz) files")
    ap.add_argument("--dir", action="append", default=[],
                    help="Directory to scan for trace files (repeatable)")
    ap.add_argument("--dir-regex", default=None,
                    help="Regex filter on immediate subdirectories of --dir")
    ap.add_argument("--trace-regex", default=r"profile_export\.jsonl(\.gz)?$",
                    help="Regex matched against filenames when discovering traces")
    ap.add_argument("--metrics", nargs="+", default=["ttft", "itl", "e2e"],
                    choices=list(METRIC_SPECS), metavar="METRIC",
                    help="Metrics to plot: ttft, itl, e2e (default: all three)")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Override auto-derived labels for each trace (in order)")
    ap.add_argument("-o", "--output", default="metric_distributions.png",
                    help="Output image path")
    ap.add_argument("--title", default=None, help="Overall figure title")
    ap.add_argument("--bins", type=int, default=200,
                    help="Number of histogram bins per subplot (default: 200)")
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="Gaussian smoothing sigma in bin units (0 = off, default: 1.0)")
    args = ap.parse_args()

    trace_paths: List[str] = list(args.traces)
    if args.dir:
        trace_paths.extend(discover_traces(args.dir, args.dir_regex, args.trace_regex))
    trace_paths = list(dict.fromkeys(trace_paths))
    if not trace_paths:
        ap.error("No trace files provided or discovered")

    labels, metric_data = collect_metrics(trace_paths, args.metrics)
    if args.labels:
        for i, lbl in enumerate(args.labels[:len(labels)]):
            labels[i] = lbl

    metric_names = [METRIC_SPECS[k][0] for k in args.metrics]
    plot_distributions(
        labels=labels,
        metric_names=metric_names,
        metric_data=metric_data,
        title=args.title,
        output=args.output,
        bins=args.bins,
        sigma=args.sigma,
    )


if __name__ == "__main__":
    main()
