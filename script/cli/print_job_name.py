#!/usr/bin/env python3
import os
import re
import sys


def sanitize_job_name(s: str, max_len: int = 128) -> str:
    s = s.strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:max_len]


def _read_top_level_keys(path: str) -> dict:
    """Very lightweight YAML top-level reader to avoid external deps.

    Only captures first-occurrence of top-level 'project' and 'name' keys,
    and stops scanning once a top-level 'parameters:' block starts.
    """
    keys = {}
    with open(path, "r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line or line.lstrip().startswith("#"):
                continue
            # stop at top-level parameters block
            if line.startswith("parameters:"):
                break
            # only consider top-level (no leading spaces)
            if line[0].isspace():
                continue
            m = re.match(r"^(\w+)\s*:\s*(.*)$", line)
            if not m:
                continue
            k, v = m.group(1), m.group(2)
            v = v.strip().strip('"\'')
            if k in ("project") and k not in keys:
                keys[k] = v
            if "project" in keys:
                break
    return keys


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: print_job_name.py <config_path>", file=sys.stderr)
        return 2
    cfg_path = sys.argv[1]
    keys = _read_top_level_keys(cfg_path)
    project = str(keys.get("project", "xai") or "xai")
    run = f"{project}"
    print(sanitize_job_name(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
