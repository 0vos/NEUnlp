import argparse
import glob
import json
import os
from typing import Any, Dict, List


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize metrics JSON files into a small table.")
    parser.add_argument("--metrics_glob", type=str, default="outputs/metrics/*.json")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.metrics_glob))
    if not paths:
        raise SystemExit(f"No metrics matched: {args.metrics_glob}")

    rows: List[Dict[str, Any]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        rows.append(
            {
                "file": os.path.basename(p),
                "model": obj.get("model_name_or_path"),
                "lora": bool(obj.get("lora_adapter_path")),
                "num": obj.get("num_samples"),
                "accuracy": obj.get("accuracy"),
            }
        )

    # Print a simple fixed-width table.
    header = ["file", "lora", "num", "accuracy", "model"]
    print("\t".join(header))
    for r in rows:
        print(
            "\t".join(
                [
                    str(r.get("file", "")),
                    str(int(bool(r.get("lora")))),
                    str(r.get("num", "")),
                    f"{float(r.get('accuracy', 0.0)):.4f}",
                    str(r.get("model", "")),
                ]
            )
        )


if __name__ == "__main__":
    main()
