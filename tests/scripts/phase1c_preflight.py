#!/usr/bin/env python3
"""Runtime preflight checks placed in `tests/scripts` and callable from sbatch.

Checks performed:
- Verify a GCC module is available (via `module list` or `gcc` on PATH).
- If CASE == 'case14', ensure `SLURM_CPUS_PER_TASK` is set and < 12.

This is intentionally lightweight and uses only the standard library.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Optional


def module_list_output() -> str:
    try:
        p = subprocess.run(["bash", "-lc", "module list 2>&1"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=10)
        return p.stdout or ""
    except Exception:
        return ""


def check_gcc_loaded() -> bool:
    """Return True if a GCC module appears in `module list` output or `gcc` is on PATH."""
    out = module_list_output().lower()
    if "gcc" in out:
        return True

    # Fallback: check if `gcc` is available on PATH
    return shutil.which("gcc") is not None


def check_case_cpus(case: Optional[str]) -> bool:
    """Enforce CPUS rule when CASE is 'case14'. Returns True if check passes.

    If `SLURM_CPUS_PER_TASK` is not set, the check is skipped (cannot evaluate).
    """
    cpus_str = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus_str is None:
        print("SLURM_CPUS_PER_TASK not set; skipping cpus check", file=sys.stderr)
        return True

    try:
        cpus = int(cpus_str)
    except Exception:
        print(f"Invalid SLURM_CPUS_PER_TASK value: {cpus_str}", file=sys.stderr)
        return False

    if case == "case14":
        if cpus < 12:
            return True
        print(f"ERROR: For CASE=case14, SLURM_CPUS_PER_TASK must be < 12 (found {cpus})", file=sys.stderr)
        return False

    return True


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Phase1c runtime preflight checks")
    p.add_argument("--case", help="Dataset CASE name (optional, can also be provided via CASE env var)")
    args = p.parse_args(argv)

    case = args.case or os.environ.get("CASE")

    ok = True

    if not check_gcc_loaded():
        print("ERROR: GCC module not detected (checked 'module list' and PATH).", file=sys.stderr)
        out = module_list_output()
        if out:
            print("-- module list output (truncated) --", file=sys.stderr)
            for line in out.splitlines()[:20]:
                print(line, file=sys.stderr)
        ok = False

    if not check_case_cpus(case):
        ok = False

    if not ok:
        return 1

    print("Preflight OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
