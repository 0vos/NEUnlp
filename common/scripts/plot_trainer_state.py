import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _extract_series(log_history: List[Dict], key: str) -> Tuple[List[int], List[float]]:
    steps: List[int] = []
    vals: List[float] = []
    for item in log_history:
        if key not in item:
            continue
        step = item.get("step")
        if step is None:
            continue
        steps.append(int(step))
        vals.append(float(item[key]))
    return steps, vals


def _plot(series: Tuple[List[int], List[float]], title: str, out_path: str) -> None:
    steps, vals = series
    if not steps:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(steps, vals)
    plt.title(title)
    plt.xlabel("step")
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot loss/lr/grad_norm (if present) from HuggingFace Trainer trainer_state.json"
    )
    parser.add_argument("--trainer_state", type=str, required=True, help="Path to trainer_state.json")
    parser.add_argument("--out_dir", type=str, default="outputs/plots")
    args = parser.parse_args()

    with open(args.trainer_state, "r", encoding="utf-8") as f:
        obj = json.load(f)
    log_history = obj.get("log_history", [])

    plots = {
        "loss": "train_loss",
        "learning_rate": "learning_rate",
        "grad_norm": "grad_norm",
    }

    made_any = False
    for key, title in plots.items():
        series = _extract_series(log_history, key)
        if series[0]:
            made_any = True
            out_path = os.path.join(args.out_dir, f"{key}.png")
            _plot(series, title, out_path)

    if not made_any:
        print("No plottable keys found in log_history. Available keys vary by trainer version.")
    else:
        print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
