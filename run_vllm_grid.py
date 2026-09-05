#!/usr/bin/env python3
"""Run the fixed-shape mechanism grid with vLLM's official serving client."""

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("scenarios/h800.json"))
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=("baremetal", "cvm"), required=True)
    p.add_argument("--result-root", type=Path, default=Path("results"))
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    scenario = next(s for s in cfg["scenarios"] if s["name"] == "mechanism-grid")
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = args.result_root / f"{run_id}-{args.mode}-mechanism-grid"
    out.mkdir(parents=True, exist_ok=True)
    commands = []
    for repeat in range(1, args.repeat + 1):
        for concurrency in scenario["concurrency"]:
            for isl in scenario["input_tokens"]:
                for osl in scenario["output_tokens"]:
                    label = f"r{repeat}-i{isl}-o{osl}-c{concurrency}"
                    cmd = [
                        "vllm", "bench", "serve", "--backend", "vllm",
                        "--endpoint", "/v1/completions", "--base-url", args.base_url,
                        "--model", args.model, "--dataset-name", "random",
                        "--random-input-len", str(isl), "--random-output-len", str(osl),
                        "--random-range-ratio", "0", "--num-prompts",
                        str(scenario["requests_per_point"]), "--request-rate", "inf",
                        "--max-concurrency", str(concurrency), "--ignore-eos",
                        "--seed", str(cfg["seed"]), "--percentile-metrics",
                        "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,95,99",
                        "--save-result", "--save-detailed", "--result-dir", str(out),
                        "--result-filename", f"{label}.json", "--metadata",
                        f"mode={args.mode}", f"repeat={repeat}", f"label={label}"
                    ]
                    commands.append(cmd)
    (out / "commands.txt").write_text("\n".join(shlex.join(c) for c in commands) + "\n")
    (out / "config.snapshot.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    for cmd in commands:
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    print(f"results: {out}")


if __name__ == "__main__":
    main()
