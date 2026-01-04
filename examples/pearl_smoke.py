# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal PEARL smoke script for local testing."""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

import torch

from vllm import AsyncEngineArgs, LLM, SamplingParams
from vllm.engine.arg_utils import human_readable_int
from vllm.v1.engine.async_llm import AsyncLLM


def _build_spec_config(args: argparse.Namespace) -> dict:
    num_spec_tokens = args.num_spec_tokens
    if num_spec_tokens is None:
        num_spec_tokens = args.gamma if args.gamma > 0 else 4

    spec_config = {
        "method": "pearl",
        "pearl_draft_model": args.draft_model,
        "pearl_target_model": args.target_model,
        "pearl_draft_tensor_parallel_size": args.draft_tp,
        "pearl_target_tensor_parallel_size": args.target_tp,
        "pearl_gamma": -1 if args.auto_gamma else args.gamma,
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

    return spec_config


def _build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    return SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=-1,
        ignore_eos=True,
        logprobs=args.logprobs if args.logprobs > 0 else None,
    )


async def _stream_outputs(
    args: argparse.Namespace,
    world_tp: int,
    spec_config: dict,
    sampling_params: SamplingParams,
) -> int:
    engine_args = AsyncEngineArgs(
        model=args.target_model,
        tensor_parallel_size=world_tp,
        speculative_config=spec_config,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        skip_kv_cache_profiling=args.skip_kv_cache_profiling,
        skip_model_warmup=args.skip_model_warmup,
        stream_interval=1,
    )

    llm = AsyncLLM.from_engine_args(engine_args)
    try:
        for idx in range(args.num_prompts):
            request_id = f"pearl-{idx}-{uuid.uuid4().hex}"
            last_text = ""
            print(f"[{idx}] ", end="", flush=True)

            async for output in llm.generate(
                args.prompt,
                sampling_params,
                request_id,
            ):
                if not output.outputs:
                    continue
                chunk = output.outputs[0].text
                if not chunk:
                    continue
                if chunk.startswith(last_text):
                    delta = chunk[len(last_text) :]
                    last_text = chunk
                else:
                    delta = chunk
                    last_text += chunk
                if delta:
                    print(delta, end="", flush=True)

            print("\n" + "-" * 40)
    finally:
        llm.shutdown()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test for PEARL speculative decoding."
    )
    parser.add_argument(
        "--target-model",
        required=True,
        help="Target model path/name (e.g. Qwen3-32B).",
    )
    parser.add_argument(
        "--draft-model",
        required=True,
        help="Draft model path/name (e.g. Qwen3-1.7B).",
    )
    parser.add_argument("--target-tp", type=int, default=1)
    parser.add_argument("--draft-tp", type=int, default=1)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--num-spec-tokens", type=int, default=None)
    parser.add_argument("--auto-gamma", action="store_true")
    parser.add_argument("--prompt", default="Give me three short facts about Taiwan.")
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--kv-cache-memory-bytes", type=human_readable_int)
    parser.add_argument(
        "--skip-kv-cache-profiling",
        action="store_true",
        help="Skip KV cache profiling (requires --kv-cache-memory-bytes).",
    )
    parser.add_argument(
        "--skip-model-warmup",
        action="store_true",
        help="Skip model warmup/compile after KV cache initialization.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--logprobs", type=int, default=0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--pearl-block-size", type=int, default=None)
    parser.add_argument("--pearl-max-num-seqs", type=int, default=None)
    parser.add_argument("--pearl-max-num-batched-tokens", type=int, default=None)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Force eager execution (disables torch.compile/cudagraph).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; PEARL requires GPU execution.")
        return 1
    world_tp = args.target_tp + args.draft_tp
    if world_tp <= 0:
        raise ValueError("Sum of target-tp and draft-tp must be positive.")
    if args.skip_kv_cache_profiling and args.kv_cache_memory_bytes is None:
        raise ValueError(
            "--skip-kv-cache-profiling requires --kv-cache-memory-bytes."
        )

    if args.max_num_batched_tokens is None and args.max_model_len is not None:
        args.max_num_batched_tokens = args.max_model_len

    spec_config = _build_spec_config(args)
    sampling_params = _build_sampling_params(args)

    if args.stream:
        return asyncio.run(
            _stream_outputs(args, world_tp, spec_config, sampling_params)
        )

    llm = LLM(
        model=args.target_model,
        tensor_parallel_size=world_tp,
        speculative_config=spec_config,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        skip_kv_cache_profiling=args.skip_kv_cache_profiling,
        skip_model_warmup=args.skip_model_warmup,
    )

    prompts = [args.prompt] * args.num_prompts
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    for idx, output in enumerate(outputs):
        if not output.outputs:
            print(f"[{idx}] empty output")
            continue
        response = output.outputs[0]
        text = response.text
        token_count = len(response.token_ids)
        print(f"[{idx}] tokens={token_count}")
        print(text)
        if response.logprobs is not None:
            print(f"[{idx}] logprobs_len={len(response.logprobs)}")
        print("-" * 40)

    return 0


if __name__ == "__main__":
    sys.exit(main())
