import argparse
import csv
import subprocess
import time
from datetime import datetime


def _query_nvidia_smi() -> dict:
    # Returns a dict for GPU 0.
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True).strip().splitlines()
    if not out:
        raise RuntimeError("nvidia-smi returned no output")
    # Only take GPU 0 by default.
    used, total, util, temp = [x.strip() for x in out[0].split(",")]
    return {
        "memory_used_mb": int(float(used)),
        "memory_total_mb": int(float(total)),
        "utilization_gpu": int(float(util)),
        "temperature_gpu": int(float(temp)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll nvidia-smi and write GPU stats to CSV.")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--interval_sec", type=float, default=1.0)
    args = parser.parse_args()

    fieldnames = [
        "timestamp",
        "memory_used_mb",
        "memory_total_mb",
        "utilization_gpu",
        "temperature_gpu",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        print(f"Writing GPU stats to {args.out_csv} every {args.interval_sec}s (Ctrl+C to stop)")
        try:
            while True:
                row = {"timestamp": datetime.now().isoformat(timespec="seconds")}
                row.update(_query_nvidia_smi())
                writer.writerow(row)
                f.flush()
                time.sleep(args.interval_sec)
        except KeyboardInterrupt:
            print("Stopped.")


if __name__ == "__main__":
    main()
