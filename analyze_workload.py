#!/usr/bin/env python3
"""Summarize a generated JSONL workload without third-party dependencies."""

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def percentile(values, p):
    values = sorted(values)
    if not values:
        return float("nan")
    k = (len(values) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] if lo == hi else values[lo] * (hi - k) + values[hi] * (k - lo)


def corr(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    with args.workload.open() as f:
        for line in f:
            row = json.loads(line)
            groups[row["scenario"]].append(row)
    for name, rows in groups.items():
        print(f"[{name}] requests={len(rows)}")
        for key in ("input_tokens", "output_tokens"):
            vals = [r[key] for r in rows]
            print(f"  {key}: min={min(vals)} p50={percentile(vals, 50):.1f} "
                  f"p90={percentile(vals, 90):.1f} p99={percentile(vals, 99):.1f} "
                  f"mean={statistics.fmean(vals):.1f} max={max(vals)}")
        xs = [r["input_tokens"] for r in rows]
        ys = [r["output_tokens"] for r in rows]
        reused = sum(r.get("prefix_tokens", 0) for r in rows)
        total = sum(xs)
        classes = Counter(r["content_class"] for r in rows)
        print(f"  pearson(ISL,OSL)={corr(xs, ys):.3f} prefix-token-ratio={reused / total:.3f}")
        print(f"  classes={dict(classes)}")


if __name__ == "__main__":
    main()
