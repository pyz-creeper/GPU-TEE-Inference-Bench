#!/usr/bin/env python3
"""Thin entrypoint for agentic replay; code lives in input_bench.agentic_runner."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from input_bench.agentic_runner import main
if __name__ == '__main__':
    raise SystemExit(main())
