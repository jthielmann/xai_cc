#!/usr/bin/env python3
import ast, re, pathlib, collections

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "script"
DOCS = ROOT / "docs"

# Buckets and labels
BUCKETS = {
    "Scripts": ["script/train.py", "script/xai.py", "script/eval.py", "script/sweep.py", "script/manifest.py"],
    "Common": ["script/common/"],
    "Data_Model": ["script/model/", "script/data_processing/"],
    "Config_Manifest": ["schema/"],
    "Outputs": [],  # static bucket (no files)
}
LABELS = {
    "script/train.py": "train.py",
    "script/xai.py": "xai.py",
    "script/eval.py": "eval.py",
    "script/sweep.py": "sweep.py",
    "script/manifest.py": "manifest.py",
}

INIT = """%%{init:{
  "securityLevel":"loose",
  "flowchart":{"htmlLabels":true,"nodeSpacing":36,"rankSpacing":72,"useMaxWidth":true},
  "themeVariables":{"fontSize":"19px"}
}}%%"""

# add near the top with INIT
INIT_BIG = """%%{init:{
  "securityLevel":"loose",
  "flowchart":{"htmlLabels":true,"nodeSpacing":44,"rankSpacing":84,"useMaxWidth":true},
  "themeVariables":{"fontSize":"22px"}
}}%%"""

def file_iter(root: pathlib.Path):
    for p in root.rglob("*.py"):
        s = str(p)
        if any(x in s for x in ("/__pycache__/", "/.venv/", "/.git/")):
            continue
        yield p

def bucket_of(path: pathlib.Path):
    s = path.as_posix()
    for b, patterns in BUCKETS.items():
        for pat in patterns:
            if pat.endswith("/") and s.startswith((ROOT / pat).as_posix()):
                return b
            if s == (ROOT / pat).as_posix():
                return b
    if "/model/" in s or "/data_processing/" in s:
        return "Data_Model"
    if "/common/" in s:
        return "Common"
    return "Scripts" if s.startswith((ROOT/"script").as_posix()) else None

def imports_in(py_file: pathlib.Path):
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except Exception:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                out.add(n.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out

def mod_to_path(mod: str):
    # map "script.common.datasets" -> script/common/datasets.py (best-effort)
    parts = mod.split(".")
    if parts[0] != "script":
        return None
    p = ROOT.joinpath(*parts)
    if (p.with_suffix(".py")).exists():
        return p.with_suffix(".py")
    if p.is_dir():
        return p
    return None

def safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s)

def main():
    DOCS.mkdir(parents=True, exist_ok=True)

    # Collect nodes with buckets
    nodes = {}  # path -> (id, bucket, label)
    by_bucket = collections.defaultdict(list)
    for p in file_iter(SRC):
        b = bucket_of(p)
        if not b:
            continue
        pid = safe_id(p.relative_to(ROOT).as_posix())
        label = LABELS.get(p.relative_to(ROOT).as_posix(), p.name)
        nodes[p] = (pid, b, label)
        by_bucket[b].append(p)

    # Build edges between files
    edges = set()
    for p, (pid, _, _) in nodes.items():
        for imp in imports_in(p):
            q = mod_to_path(imp)
            if q and q in nodes:
                qid, _, _ = nodes[q]
                edges.add((pid, qid))

    # ---- 1) Write BUCKET OVERVIEW (edges aggregated by bucket) ----
    bucket_edges = set()
    for p, (pid, b1, _) in nodes.items():
        for imp in imports_in(p):
            q = mod_to_path(imp)
            if q and q in nodes:
                _, b2, _ = nodes[q]
                if b1 != b2:
                    bucket_edges.add((b1, b2))

    overview = [INIT, "flowchart TB", ""]
    order = ["Scripts", "Common", "Data_Model", "Config_Manifest", "Outputs"]
    for b in order:
        overview.append(f'{b}["{b.replace("_", " ")}"];')
    overview.append("")
    for a, b in sorted(bucket_edges):
        overview.append(f"{a} --> {b};")
    # light layout hint
    overview += ["", "Scripts -.-> Common;", "Common -.-> Data_Model;", "Data_Model -.-> Config_Manifest;", "Config_Manifest -.-> Outputs;"]
    (DOCS / "_arch_overview.mmd").write_text("\n".join(overview) + "\n", encoding="utf-8")

    # ---- 2) Write PER-BUCKET FILE GRAPHS (only within-bucket edges) ----
    for b in order:
        # Use bigger init just for Scripts
        init = INIT_BIG if b == "Scripts" else INIT
        lines = [init, "flowchart TB", f"subgraph {b} [{b.replace('_',' ')}]", "  direction TB"]
        files = sorted(by_bucket.get(b, []), key=lambda p: nodes[p][0])
        bucket_ids = []
        for p in files:
            pid, _, label = nodes[p]
            label = label.replace("(", "").replace(")", "").replace(",", " ")
            lines.append(f"  {pid}[{label}];")
            bucket_ids.append(pid)
        lines.append("end\n")
        # within-bucket edges only
        for a, c in sorted(edges):
            pa = next((pp for pp, (pid, bb, _) in nodes.items() if pid == a and bb == b), None)
            pc = next((pp for pp, (pid, bb, _) in nodes.items() if pid == c and bb == b), None)
            if pa and pc:
                lines.append(f"{a} --> {c};")

        # Make nodes even bigger via class (only for Scripts)
        if b == "Scripts" and bucket_ids:
            lines.append("classDef big fill:#fff,stroke:#999,stroke-width:1px,font-size:24px;")
            lines.append(f"class {','.join(bucket_ids)} big;")

        (DOCS / f"_arch_files_{b}.mmd").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- 3) Write a ready-to-serve page that embeds all blocks ----
    page = [
        "# Project Architecture — Auto-Generated",
        "",
        "## Overview (modules)",
        "```mermaid",
        (DOCS / "_arch_overview.mmd").read_text(encoding="utf-8").strip(),
        "```",
    ]
    for b in order:
        if (DOCS / f"_arch_files_{b}.mmd").exists():
            page += [
                "",
                f"## {b.replace('_',' ')} (files)",
                "```mermaid",
                (DOCS / f"_arch_files_{b}.mmd").read_text(encoding="utf-8").strip(),
                "```",
            ]
    (DOCS / "architecture_auto.md").write_text("\n".join(page) + "\n", encoding="utf-8")

    print("Wrote:")
    print(" - docs/_arch_overview.mmd")
    for b in order:
        p = DOCS / f"_arch_files_{b}.mmd"
        if p.exists():
            print(f" - {p.name}")
    print(" - docs/architecture_auto.md")

if __name__ == "__main__":
    main()