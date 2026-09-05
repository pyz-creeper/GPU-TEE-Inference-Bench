#!/usr/bin/env python3
"""Generate deterministic, content-free workload manifests.

The output describes workload shape; a framework adapter is responsible for
materializing prompts with the exact tokenizer. This separation prevents a
tokenizer/template change from silently changing an A/B workload.
"""

import argparse
import json
import math
import random
from pathlib import Path


def weighted_choice(rng, buckets):
    x = rng.random() * sum(float(b["weight"]) for b in buckets)
    for bucket in buckets:
        x -= float(bucket["weight"])
        if x <= 0:
            return bucket
    return buckets[-1]


def log_uniform_int(rng, bounds):
    low, high = map(int, bounds)
    if low == high:
        return low
    return max(low, min(high, round(math.exp(rng.uniform(math.log(low), math.log(high))))))


def interarrival(rng, arrival):
    if arrival.get("type") in (None, "closed_loop"):
        return 0.0
    rate = float(arrival["rate_rps"])
    if arrival["type"] == "poisson":
        return rng.expovariate(rate)
    if arrival["type"] == "gamma":
        shape = float(arrival["shape"])
        return rng.gammavariate(shape, 1.0 / (rate * shape))
    raise ValueError(f"unknown arrival type: {arrival['type']}")


def record(name, request_id, scheduled, isl, osl, concurrency, prefix_tokens=0,
           prefix_group=None, content_class="synthetic"):
    return {
        "scenario": name,
        "request_id": f"{name}-{request_id:07d}",
        "scheduled_time_s": round(scheduled, 9),
        "input_tokens": int(isl),
        "output_tokens": int(osl),
        "concurrency": int(concurrency),
        "content_class": content_class,
        "prefix_tokens": int(prefix_tokens),
        "prefix_group": prefix_group,
        "sampling": {"temperature": 0.0, "ignore_eos": True}
    }


def generate_scenario(rng, scenario):
    name = scenario["name"]
    prefix = scenario.get("prefix", {})
    scheduled = 0.0
    out = []
    if scenario["kind"] == "grid":
        rid = 0
        for concurrency in scenario["concurrency"]:
            for isl in scenario["input_tokens"]:
                for osl in scenario["output_tokens"]:
                    for _ in range(int(scenario["requests_per_point"])):
                        out.append(record(name, rid, scheduled, isl, osl, concurrency))
                        rid += 1
        return out
    if scenario["kind"] != "mixture":
        raise ValueError(f"unknown scenario kind: {scenario['kind']}")
    concurrencies = list(map(int, scenario["concurrency"]))
    for rid in range(int(scenario["requests"])):
        bucket = weighted_choice(rng, scenario["length_buckets"])
        isl = log_uniform_int(rng, bucket["input"])
        # Sampling from the same bucket preserves an explicit task-conditioned
        # ISL/OSL relation instead of assuming the two are globally independent.
        osl = log_uniform_int(rng, bucket["output"])
        scheduled += interarrival(rng, scenario.get("arrival", {}))
        group = None
        shared = 0
        if rng.random() < float(prefix.get("ratio", 0.0)):
            group = f"p{rng.randrange(int(prefix.get('groups', 1))):04d}"
            shared = min(isl, log_uniform_int(rng, prefix.get("tokens", [1, 1])))
        out.append(record(name, rid, scheduled, isl, osl,
                          rng.choice(concurrencies), shared, group,
                          bucket.get("class", "synthetic")))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scenario", action="append", help="only generate named scenario")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rng = random.Random(int(config["seed"]))
    selected = set(args.scenario or [])
    records = []
    for scenario in config["scenarios"]:
        if not selected or scenario["name"] in selected:
            records.extend(generate_scenario(rng, scenario))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for item in records:
            f.write(json.dumps(item, sort_keys=True) + "\n")
    print(f"wrote {len(records)} requests to {args.output}")


if __name__ == "__main__":
    main()
