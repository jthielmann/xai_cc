#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        import jsonschema  # type: ignore
    except Exception:  # pragma: no cover
        print(
            "jsonschema is required. Install with: conda install -c conda-forge jsonschema",
            file=sys.stderr,
        )
        return 2

    manifest_path = Path(args.path).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    schema_path = (
        Path(args.schema).resolve() if args.schema else repo_root / "schema" / "manifest.v1.json"
    )

    try:
        manifest = load_json(manifest_path)
    except Exception as e:
        print(f"Failed to read manifest: {e}", file=sys.stderr)
        return 2

    try:
        schema = load_json(schema_path)
    except Exception as e:
        print(f"Failed to read schema: {e}", file=sys.stderr)
        return 2

    try:
        jsonschema.validate(instance=manifest, schema=schema)
    except jsonschema.ValidationError as e:  # type: ignore
        print("Manifest invalid:", file=sys.stderr)
        print(f" - {e.message}", file=sys.stderr)
        # Optionally show path in the manifest
        if e.path:
            print(f" - at: {'/'.join(map(str, e.path))}", file=sys.stderr)
        return 2

    print("Manifest is valid against schema.")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    manifest_path = Path(args.path).resolve()
    try:
        manifest = load_json(manifest_path)
    except Exception as e:
        print(f"Failed to read manifest: {e}", file=sys.stderr)
        return 2

    # Print a small summary
    fields = {
        "run_id": manifest.get("run_id"),
        "created_at": manifest.get("created_at"),
        "dataset": manifest.get("dataset", {}).get("name"),
        "encoder": manifest.get("encoder", {}).get("arch"),
        "checkpoints": len(manifest.get("checkpoints", []) or []),
    }
    print(json.dumps(fields, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Manifest utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="Validate a manifest against the v1 schema")
    p_val.add_argument("path", help="Path to manifest.json")
    p_val.add_argument("--schema", help="Path to schema JSON (defaults to schema/manifest.v1.json)")
    p_val.set_defaults(func=cmd_validate)

    p_show = sub.add_parser("show", help="Show a brief summary of a manifest")
    p_show.add_argument("path", help="Path to manifest.json")
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

