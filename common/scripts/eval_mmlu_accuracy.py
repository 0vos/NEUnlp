import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Example:
    subject: str
    instruction: str
    input: str
    output: str


def _load_jsonl(path: str, max_samples: int = 0) -> List[Example]:
    items: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append(
                Example(
                    subject=obj.get("subject", "unknown"),
                    instruction=obj["instruction"],
                    input=obj["input"],
                    output=obj["output"],
                )
            )
            if max_samples > 0 and len(items) >= max_samples:
                break
    return items


def _build_prompt(ex: Example) -> str:
    # Keep it simple and consistent with prepare_mmlu.
    return f"{ex.instruction}\n\n{ex.input}\n\nAnswer:"


def _logprob_of_continuation(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
    cont_lens: torch.Tensor,
) -> torch.Tensor:
    """Compute sum log-prob of continuation tokens for each sequence in batch.

    input_ids = prompt_ids + cont_ids (padded)
    prompt_lens: length of prompt for each item
    cont_lens: length of continuation for each item
    """
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, T, V]
        log_probs = torch.log_softmax(logits, dim=-1)

    batch_size, seq_len, _ = log_probs.shape
    # token t is predicted at position t-1
    # continuation token indices start at prompt_len ... prompt_len+cont_len-1
    total = torch.zeros(batch_size, device=input_ids.device, dtype=torch.float32)

    for b in range(batch_size):
        p_len = int(prompt_lens[b].item())
        c_len = int(cont_lens[b].item())
        if c_len <= 0:
            continue
        for i in range(c_len):
            token_pos = p_len + i
            pred_pos = token_pos - 1
            token_id = int(input_ids[b, token_pos].item())
            total[b] += log_probs[b, pred_pos, token_id]
    return total


def _batched(iterable: List[Example], batch_size: int) -> List[List[Example]]:
    return [iterable[i : i + batch_size] for i in range(0, len(iterable), batch_size)]


def _configure_hf_http_backend(connect_timeout_s: float, read_timeout_s: float) -> None:
    """Configure HF Hub downloads to use a more tolerant HTTP client.

    This helps in environments where TLS handshakes or large shard downloads can
    intermittently time out.
    """

    try:
        import httpx
        from huggingface_hub import configure_http_backend
    except Exception:
        return

    timeout = httpx.Timeout(
        connect=connect_timeout_s,
        read=read_timeout_s,
        write=read_timeout_s,
        pool=read_timeout_s,
    )

    def _factory() -> "httpx.Client":
        return httpx.Client(timeout=timeout, follow_redirects=True)

    configure_http_backend(_factory)


def _retry_call(fn, *, desc: str, retries: int, backoff_seconds: float):
    last_exc: Optional[BaseException] = None
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - network stack raises many types
            last_exc = exc
            if attempt >= attempts:
                break
            sleep_s = backoff_seconds * (2 ** (attempt - 1))
            print(
                f"[WARN] {desc} failed (attempt {attempt}/{attempts}): "
                f"{type(exc).__name__}: {exc}"
            )
            print(f"[WARN] Retrying after {sleep_s:.1f}s...")
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MMLU accuracy via log-likelihood scoring of A/B/C/D.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--eval_jsonl", type=str, required=True)
    parser.add_argument("--lora_adapter_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--save_json", type=str, default="")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to Transformers (needed for some repos).",
    )
    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default="",
        help="Optional HuggingFace Hub endpoint, e.g. https://hf-mirror.com (also via HF_ENDPOINT env).",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default="",
        help="Optional HF token (sets HF_TOKEN/HUGGINGFACE_HUB_TOKEN).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Optional cache dir for Transformers/HF Hub downloads.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Do not attempt any network access; use only local cache.",
    )
    parser.add_argument(
        "--download_retries",
        type=int,
        default=3,
        help="Retry count for tokenizer/model loading (useful for flaky networks).",
    )
    parser.add_argument(
        "--retry_backoff",
        type=float,
        default=2.0,
        help="Initial backoff seconds for retries (exponential).",
    )
    parser.add_argument(
        "--hf_max_workers",
        type=int,
        default=1,
        help="Set HF_HUB_MAX_WORKERS to reduce concurrent shard downloads (more stable on slow TLS).",
    )
    parser.add_argument(
        "--hf_connect_timeout",
        type=float,
        default=60.0,
        help="HTTP connect/TLS handshake timeout seconds for HF Hub.",
    )
    parser.add_argument(
        "--hf_read_timeout",
        type=float,
        default=600.0,
        help="HTTP read timeout seconds for HF Hub large file downloads.",
    )
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.hf_token:
        os.environ.setdefault("HF_TOKEN", args.hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", args.hf_token)
    if args.hf_max_workers > 0:
        os.environ["HF_HUB_MAX_WORKERS"] = str(args.hf_max_workers)
    if args.local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    _configure_hf_http_backend(args.hf_connect_timeout, args.hf_read_timeout)

    examples = _load_jsonl(args.eval_jsonl, max_samples=args.max_samples)
    if not examples:
        raise RuntimeError(f"No examples loaded from {args.eval_jsonl}")

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    tokenizer = _retry_call(
        lambda: AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=True,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir or None,
            local_files_only=args.local_files_only,
        ),
        desc="AutoTokenizer.from_pretrained",
        retries=args.download_retries,
        backoff_seconds=args.retry_backoff,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Left padding makes it easier to slice the true prompt tokens.
    tokenizer.padding_side = "left"

    model = _retry_call(
        lambda: AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch_dtype if args.device.startswith("cuda") else None,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir or None,
            local_files_only=args.local_files_only,
        ),
        desc="AutoModelForCausalLM.from_pretrained",
        retries=args.download_retries,
        backoff_seconds=args.retry_backoff,
    )
    model = model.to(args.device)
    model.eval()

    if args.lora_adapter_path:
        model = PeftModel.from_pretrained(model, args.lora_adapter_path)
        model = model.to(args.device)
        model.eval()

    letters = ["A", "B", "C", "D"]
    # Pre-tokenize option continuations. Prefix with a space for most tokenizers.
    option_texts = [" " + l for l in letters]
    option_ids = [tokenizer.encode(t, add_special_tokens=False) for t in option_texts]

    correct = 0
    total = 0
    by_subject: Dict[str, Tuple[int, int]] = {}

    batches = _batched(examples, args.batch_size)
    for batch in tqdm(batches, desc="Eval"):
        prompts = [_build_prompt(ex) for ex in batch]
        prompt_enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        prompt_input_ids = prompt_enc["input_ids"]
        prompt_attn = prompt_enc["attention_mask"]
        prompt_lens = prompt_attn.sum(dim=1)

        # Expand each prompt into 4 options.
        expanded_input_ids: List[torch.Tensor] = []
        expanded_attn: List[torch.Tensor] = []
        expanded_prompt_lens: List[int] = []
        expanded_cont_lens: List[int] = []
        group_map: List[int] = []  # map expanded row -> original index in batch
        option_map: List[int] = []  # which option index

        for i in range(len(batch)):
            p_len = int(prompt_lens[i].item())
            # Remove left padding: keep only tokens where attention_mask==1
            p_ids = prompt_input_ids[i, -p_len:]
            for opt_i, opt in enumerate(option_ids):
                cont = torch.tensor(opt, dtype=torch.long)
                seq = torch.cat([p_ids, cont], dim=0)
                expanded_input_ids.append(seq)
                expanded_attn.append(torch.ones_like(seq))
                expanded_prompt_lens.append(int(p_ids.shape[0]))
                expanded_cont_lens.append(int(cont.shape[0]))
                group_map.append(i)
                option_map.append(opt_i)

        # Pad expanded sequences
        max_len = max(int(x.shape[0]) for x in expanded_input_ids)
        pad_id = tokenizer.pad_token_id
        input_ids = torch.full((len(expanded_input_ids), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(expanded_input_ids), max_len), dtype=torch.long)
        for r, seq in enumerate(expanded_input_ids):
            input_ids[r, : seq.shape[0]] = seq
            attention_mask[r, : seq.shape[0]] = 1

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        prompt_lens_t = torch.tensor(expanded_prompt_lens, device=device)
        cont_lens_t = torch.tensor(expanded_cont_lens, device=device)

        scores = _logprob_of_continuation(model, input_ids, attention_mask, prompt_lens_t, cont_lens_t)

        # Aggregate scores back to per-example.
        per_ex_scores = torch.full((len(batch), 4), -1e9, device=device, dtype=torch.float32)
        for row, sc in enumerate(scores):
            per_ex_scores[group_map[row], option_map[row]] = sc

        preds = per_ex_scores.argmax(dim=1).tolist()

        for i, ex in enumerate(batch):
            total += 1
            gold = ex.output.strip().upper()
            pred_letter = letters[preds[i]]
            is_correct = pred_letter == gold
            if is_correct:
                correct += 1
            subj = ex.subject
            c, t = by_subject.get(subj, (0, 0))
            by_subject[subj] = (c + (1 if is_correct else 0), t + 1)

    acc = correct / max(1, total)
    result = {
        "eval_jsonl": args.eval_jsonl,
        "model_name_or_path": args.model_name_or_path,
        "lora_adapter_path": args.lora_adapter_path or None,
        "num_samples": total,
        "accuracy": acc,
        "by_subject": {k: (v[0] / v[1]) for k, v in sorted(by_subject.items())},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved: {args.save_json}")


if __name__ == "__main__":
    main()
