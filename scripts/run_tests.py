#!/usr/bin/env python3
"""Run CPU/mock suites in separate interpreters to isolate vLLM import stubs."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    for name in ('test_retention.py', 'test_installer.py', 'test_comparison.py',
                 'test_smoke.py', 'test_benchmark.py', 'test_diagnostics.py'):
        print(f'\n=== {name} ===', flush=True)
        subprocess.run([sys.executable, str(ROOT/'tests'/name)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT/'scripts/summarize_results.py')], cwd=ROOT, check=True)
