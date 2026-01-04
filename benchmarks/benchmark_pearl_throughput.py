# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PEARL throughput/concurrency benchmark (offline)."""

from __future__ import annotations

import argparse
import math
import random
import time
from typing import Iterable

import torch

from vllm import LLM, SamplingParams
from vllm.inputs import token_inputs


def _parse_int_list(value: str) -> list[int]:
    items = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        items.append(int(chunk))
    if not items:
        raise ValueError("Expected at least one integer value.")
    return items


def _encode_prompt_ids(tokenizer, prompt_len: int, text: str) -> list[int]:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not token_ids:
        token_ids = [1]
    if prompt_len <= len(token_ids):
        return token_ids[:prompt_len]
    repeats = math.ceil(prompt_len / len(token_ids))
    return (token_ids * repeats)[:prompt_len]


def _random_prompt_ids(
    tokenizer, prompt_len: int, seed: int, avoid_token: int | None = None
) -> list[int]:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab_size = len(tokenizer)
    if vocab_size <= 2:
        return [1] * prompt_len
    rng = random.Random(seed)
    start = 1
    end = vocab_size - 1
    ids = []
    for _ in range(prompt_len):
        token_id = rng.randint(start, end)
        if avoid_token is not None and token_id == avoid_token:
            token_id = start
        ids.append(token_id)
    return ids


def _batched(iterable: Iterable, batch_size: int) -> Iterable[list]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _count_output_tokens(outputs) -> int:
    total = 0
    for output in outputs:
        if not output.outputs:
            continue
        total += len(output.outputs[0].token_ids)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline PEARL throughput/concurrency benchmark."
    )
    parser.add_argument("--model", required=True, help="Target model name/path.")
    parser.add_argument(
        "--draft-model", required=True, help="Draft model name/path."
    )
    parser.add_argument("--target-tp", type=int, default=1)
    parser.add_argument("--draft-tp", type=int, default=1)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--num-spec-tokens", type=int, default=4)
    parser.add_argument("--auto-gamma", action="store_true")
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument(
        "--concurrency",
        type=_parse_int_list,
        default=[1, 2, 4, 8],
        help="Comma-separated concurrency list.",
    )
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--pearl-block-size", type=int, default=None)
    parser.add_argument("--pearl-max-num-seqs", type=int, default=None)
    parser.add_argument("--pearl-max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--prompt-text", default="hello")
    parser.add_argument("--random-prompts", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    num_spec_tokens = args.num_spec_tokens
    if args.gamma > 0 and not args.auto_gamma:
        num_spec_tokens = args.gamma
    spec_config = {
        "method": "pearl",
        "pearl_draft_model": args.draft_model,
        "pearl_target_model": args.model,
        "pearl_draft_tensor_parallel_size": args.draft_tp,
        "pearl_target_tensor_parallel_size": args.target_tp,
        "pearl_gamma": args.gamma if not args.auto_gamma else -1,
        "pearl_auto_gamma": args.auto_gamma,
        "num_speculative_tokens": num_spec_tokens,
    }
    if args.pearl_block_size is not None:
        spec_config["pearl_block_size"] = args.pearl_block_size
    if args.pearl_max_num_seqs is not None:
        spec_config["pearl_max_num_seqs"] = args.pearl_max_num_seqs
    if args.pearl_max_num_batched_tokens is not None:
        spec_config["pearl_max_num_batched_tokens"] = (
            args.pearl_max_num_batched_tokens
        )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.target_tp,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config=spec_config,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )

    tokenizer = llm.get_tokenizer()
    if args.random_prompts:
        prompt_ids = _random_prompt_ids(
            tokenizer, args.prompt_len, args.seed, avoid_token=0
        )
    else:
        prompt_ids = _encode_prompt_ids(
            tokenizer, args.prompt_len, args.prompt_text
        )

    sampling_params = SamplingParams(
        max_tokens=args.output_len,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        ignore_eos=True,
    )

    print("PEARL benchmark configuration")
    print(f"model: {args.model}")
    print(f"draft_model: {args.draft_model}")
    print(f"prompt_len: {args.prompt_len}")
    print(f"output_len: {args.output_len}")
    print(f"num_prompts: {args.num_prompts}")
    print(f"concurrency: {args.concurrency}")
    print("")

    for concurrency in args.concurrency:
        num_prompts = args.num_prompts
        def make_prompts() -> list[dict]:
            return [token_inputs(prompt_ids.copy()) for _ in range(num_prompts)]

        for _ in range(args.warmup_iters):
            for batch in _batched(make_prompts(), concurrency):
                _ = llm.generate(batch, sampling_params, use_tqdm=False)

        prompts = make_prompts()
        start = time.perf_counter()
        total_out_tokens = 0
        total_reqs = 0
        for batch in _batched(prompts, concurrency):
            outputs = llm.generate(batch, sampling_params, use_tqdm=False)
            total_out_tokens += _count_output_tokens(outputs)
            total_reqs += len(outputs)
        elapsed = time.perf_counter() - start
        prompt_tokens = total_reqs * args.prompt_len
        total_tokens = prompt_tokens + total_out_tokens

        print(f"concurrency={concurrency}")
        print(f"  elapsed_s: {elapsed:.4f}")
        print(f"  requests: {total_reqs}")
        print(f"  output_tokens: {total_out_tokens}")
        print(f"  tokens_per_s_output: {total_out_tokens / elapsed:.2f}")
        print(f"  tokens_per_s_total: {total_tokens / elapsed:.2f}")
        print(f"  requests_per_s: {total_reqs / elapsed:.2f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
