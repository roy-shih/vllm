# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal PEARL streaming smoke script using AsyncLLM."""

from __future__ import annotations

import argparse
import asyncio
import sys

import torch

from vllm import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.utils import random_uuid
from vllm.v1.engine.async_llm import AsyncLLM


def _build_spec_config(args: argparse.Namespace) -> dict:
    num_spec_tokens = args.num_spec_tokens
    if num_spec_tokens is None:
        num_spec_tokens = args.gamma if args.gamma > 0 else 4

    return {
        "method": "pearl",
        "pearl_draft_model": args.draft_model,
        "pearl_target_model": args.target_model,
        "pearl_draft_tensor_parallel_size": args.draft_tp,
        "pearl_target_tensor_parallel_size": args.target_tp,
        "pearl_gamma": -1 if args.auto_gamma else args.gamma,
        "pearl_auto_gamma": args.auto_gamma,
        "num_speculative_tokens": num_spec_tokens,
    }


async def _run(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available; PEARL requires GPU execution.")
        return 1
    world_tp = args.target_tp + args.draft_tp
    if world_tp <= 0:
        raise ValueError("Sum of target-tp and draft-tp must be positive.")

    engine_args = AsyncEngineArgs(
        model=args.target_model,
        tensor_parallel_size=world_tp,
        speculative_config=_build_spec_config(args),
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        disable_log_stats=True,
        stream_interval=args.stream_interval,
        enforce_eager=args.enforce_eager,
    )
    llm = AsyncLLM.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=-1,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
    )

    request_id = random_uuid()
    print("streaming output:")
    try:
        async for output in llm.generate(
            args.prompt, sampling_params, request_id=request_id
        ):
            if not output.outputs:
                continue
            delta = output.outputs[0].text
            if delta:
                print(delta, end="", flush=True)
            if output.finished:
                print()
                break
    finally:
        llm.shutdown()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Streaming PEARL smoke test using AsyncLLM."
    )
    parser.add_argument(
        "--target-model",
        required=True,
        help="Target model path/name (e.g. Qwen3-1.7B).",
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
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Force eager execution (disables torch.compile/cudagraph).",
    )
    args = parser.parse_args()

    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
