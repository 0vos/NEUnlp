import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

from tqdm import tqdm


@dataclass
class MMLUSample:
    subject: str
    question: str
    choices: List[str]
    answer_index: int


def _format_input(question: str, choices: List[str]) -> str:
    letters = ["A", "B", "C", "D"]
    lines = ["Question:", question.strip(), "", "Choices:"]
    for i, choice in enumerate(choices):
        prefix = letters[i] if i < len(letters) else str(i)
        lines.append(f"{prefix}. {str(choice).strip()}")
    lines.append("\nAnswer with a single letter: A, B, C, or D.")
    return "\n".join(lines)


def _to_alpaca_record(sample: MMLUSample) -> dict:
    letters = ["A", "B", "C", "D"]
    answer_letter = letters[sample.answer_index]
    return {
        "instruction": (
            "You are given a multiple-choice question. "
            "Choose the correct option. Respond with only the letter (A, B, C, or D)."
        ),
        "input": _format_input(sample.question, sample.choices),
        "output": answer_letter,
        "subject": sample.subject,
    }


def _iter_subjects(subjects_arg: str) -> List[str]:
    subjects_arg = subjects_arg.strip()
    if subjects_arg.lower() == "all":
        from datasets import get_dataset_config_names

        return sorted(get_dataset_config_names("cais/mmlu"))
    return [s.strip() for s in subjects_arg.split(",") if s.strip()]


def _load_dataset_with_retries(
    subject: str,
    split: str,
    retries: int,
    backoff_seconds: float,
) -> "object":
    from datasets import load_dataset

    last_exc: Optional[BaseException] = None
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            return load_dataset("cais/mmlu", subject, split=split)
        except Exception as exc:  # noqa: BLE001 - intentionally broad for network stack variety
            # Deterministic configuration errors should not be retried.
            if isinstance(exc, ValueError) and "Unknown split" in str(exc):
                try:
                    ds_dict = load_dataset("cais/mmlu", subject)
                    available = sorted(list(ds_dict.keys()))
                except Exception:
                    available = []

                hint = (
                    "The cais/mmlu dataset config you are using does not provide an "
                    "'auxiliary_train' split for per-subject configs. In many setups, "
                    "you should use '--train_split dev' and '--eval_split test'."
                )
                raise RuntimeError(
                    f"Unknown split '{split}' for cais/mmlu subject='{subject}'. "
                    + (f"Available splits: {available}. " if available else "")
                    + hint
                ) from exc

            last_exc = exc
            if attempt >= attempts:
                break
            sleep_s = backoff_seconds * (2 ** (attempt - 1))
            print(
                f"[WARN] load_dataset cais/mmlu:{subject} split={split} failed "
                f"(attempt {attempt}/{attempts}): {type(exc).__name__}: {exc}"
            )
            print(f"[WARN] Retrying after {sleep_s:.1f}s...")
            time.sleep(sleep_s)

    assert last_exc is not None
    raise RuntimeError(
        "Failed to download/load cais/mmlu after retries. "
        "If you are behind a slow/blocked network, try setting a mirror endpoint, e.g. "
        "`--hf_endpoint https://hf-mirror.com` (or export HF_ENDPOINT)."
    ) from last_exc


def _load_split(subject: str, split: str, retries: int, backoff_seconds: float) -> Iterable[MMLUSample]:
    ds = _load_dataset_with_retries(subject=subject, split=split, retries=retries, backoff_seconds=backoff_seconds)
    for row in ds:
        yield MMLUSample(
            subject=subject,
            question=row["question"],
            choices=list(row["choices"]),
            answer_index=int(row["answer"]),
        )


def _write_jsonl(path: str, records: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cais/mmlu into Alpaca-style JSONL.")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--train_split", type=str, default="dev")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--subjects", type=str, default="all", help="'all' or comma-separated subject list")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train", type=int, default=0, help="0 means no limit")
    parser.add_argument("--max_eval", type=int, default=0, help="0 means no limit")
    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default="",
        help="Optional HuggingFace Hub endpoint (sets HF_ENDPOINT env var), e.g. https://hf-mirror.com",
    )
    parser.add_argument("--retries", type=int, default=3, help="Retry count for HF dataset download")
    parser.add_argument("--retry_backoff", type=float, default=2.0, help="Initial backoff seconds (exponential)")
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    random.seed(args.seed)
    subjects = _iter_subjects(args.subjects)

    train_records: List[dict] = []
    eval_records: List[dict] = []
    train_target = None if args.max_train <= 0 else args.max_train
    eval_target = None if args.max_eval <= 0 else args.max_eval

    for subject in tqdm(subjects, desc="Subjects"):
        if train_target is None or len(train_records) < train_target:
            for sample in _load_split(subject, args.train_split, retries=args.retries, backoff_seconds=args.retry_backoff):
                train_records.append(_to_alpaca_record(sample))
                if train_target is not None and len(train_records) >= train_target:
                    break

        if eval_target is None or len(eval_records) < eval_target:
            for sample in _load_split(subject, args.eval_split, retries=args.retries, backoff_seconds=args.retry_backoff):
                eval_records.append(_to_alpaca_record(sample))
                if eval_target is not None and len(eval_records) >= eval_target:
                    break

        if train_target is not None and eval_target is not None:
            if len(train_records) >= train_target and len(eval_records) >= eval_target:
                break

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "mmlu_train.jsonl")
    eval_path = os.path.join(out_dir, "mmlu_eval.jsonl")
    meta_path = os.path.join(out_dir, "mmlu_meta.json")
    dataset_info_path = os.path.join(out_dir, "dataset_info.json")

    _write_jsonl(train_path, train_records)
    _write_jsonl(eval_path, eval_records)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "cais/mmlu",
                "subjects": subjects,
                "seed": args.seed,
                "train_split": args.train_split,
                "eval_split": args.eval_split,
                "num_train": len(train_records),
                "num_eval": len(eval_records),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # LLaMA-Factory: register custom datasets via dataset_info.json under dataset_dir.
    # See: https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md
    dataset_info = {
        "mmlu_mcq_train": {
            "file_name": "mmlu_train.jsonl",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        "mmlu_mcq_eval": {
            "file_name": "mmlu_eval.jsonl",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
    }
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print(f"Wrote train: {train_path} ({len(train_records)})")
    print(f"Wrote eval : {eval_path} ({len(eval_records)})")
    print(f"Wrote meta : {meta_path}")
    print(f"Wrote info : {dataset_info_path}")


if __name__ == "__main__":
    main()
