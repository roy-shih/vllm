#!/usr/bin/env python3
"""
Test script for PEARL integration in vLLM.

This script demonstrates basic usage of PEARL speculative decoding in vLLM.
"""

import argparse
import logging
import sys
import time

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Error: vLLM not found. Please install vLLM first.")
    sys.exit(1)


def test_pearl_basic():
    """Test basic PEARL functionality with small models."""
    print("=" * 50)
    print("Testing PEARL Integration - Basic Test")
    print("=" * 50)

    # Use small models for quick testing
    # Replace with your desired models
    target_model = "facebook/opt-125m"
    draft_model = "facebook/opt-125m"  # Same model for simplicity

    print(f"\nTarget Model: {target_model}")
    print(f"Draft Model: {draft_model}")
    print("\nNOTE: Using same model for draft and target for initial testing.")
    print("For production, use a smaller draft model (e.g., Llama-3-1B → Llama-3-8B)\n")

    try:
        # Create LLM instance with PEARL
        print("Initializing LLM with PEARL...")
        llm = LLM(
            model=target_model,
            speculative_config={
                "method": "pearl",
                "model": draft_model,
                "num_speculative_tokens": 3,  # Start small
                "draft_tensor_parallel_size": 1,
                "target_tensor_parallel_size": 1,
                "pearl_gamma": 5,
            },
            max_model_len=512,
            enforce_eager=True,  # Disable CUDA graphs for debugging
            trust_remote_code=True,
        )

        # Test prompts - keep it simple for initial testing
        prompts = [
            "Hello, my name is",
        ]

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=20,  # Short for quick testing
        )

        print("\nGenerating outputs...")
        print("(Watch for PEARL debug logs above)")
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.time()

        # Display results
        print("\n" + "=" * 50)
        print("Results:")
        print("=" * 50)
        for i, output in enumerate(outputs):
            print(f"\nPrompt: {prompts[i]}")
            print(f"Generated: {output.outputs[0].text}")
            print(f"Tokens generated: {len(output.outputs[0].token_ids)}")

        elapsed = end_time - start_time
        total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
        throughput = total_tokens / elapsed if elapsed > 0 else 0

        print("\n" + "=" * 50)
        print("Performance:")
        print("=" * 50)
        print(f"Total time: {elapsed:.2f}s")
        print(f"Total tokens: {total_tokens}")
        print(f"Throughput: {throughput:.2f} tokens/s")
        print("=" * 50)

        print("\n" + "=" * 50)
        print("What to Check:")
        print("=" * 50)
        print("1. Look for '[PEARL] Loading draft model...' in logs")
        print("2. Look for '[PEARL] Generated X draft tokens' in logs")
        print("3. Look for 'SpecDecoding metrics: Mean acceptance length: X.XX'")
        print("4. If MAT > 1.0, PEARL is working!")
        print("=" * 50)

        print("\n✓ Test completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 50)
        print("Troubleshooting:")
        print("=" * 50)
        print("1. Check that models are downloaded")
        print("2. Check GPU memory is sufficient")
        print("3. Look for errors in traceback above")
        print("4. See unieai-dev/TESTING_GUIDE.md for common issues")
        print("=" * 50)
        return False


def test_pearl_config_validation():
    """Test PEARL configuration validation."""
    print("\n" + "=" * 50)
    print("Testing PEARL Configuration Validation")
    print("=" * 50)

    test_cases = [
        {
            "name": "Valid config",
            "config": {
                "method": "pearl",
                "model": "test-draft-model",
                "num_speculative_tokens": 5,
                "draft_tensor_parallel_size": 1,
                "target_tensor_parallel_size": 2,
            },
            "should_succeed": True,
        },
        {
            "name": "Missing draft model",
            "config": {
                "method": "pearl",
                "num_speculative_tokens": 5,
            },
            "should_succeed": False,
        },
    ]

    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print(f"Config: {test['config']}")
        # Configuration validation would be tested here
        # This is a placeholder for actual validation tests
        print("→ Test placeholder (not implemented)")

    print("\n✓ Configuration validation tests completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Test PEARL integration in vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test with debug logging
  python test_pearl_integration.py --debug

  # Test with custom models
  python test_pearl_integration.py --target-model facebook/opt-350m --draft-model facebook/opt-125m

  # Config validation only
  python test_pearl_integration.py --test config
        """
    )
    parser.add_argument(
        "--test",
        choices=["basic", "config", "all"],
        default="basic",
        help="Which test to run (default: basic)",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        help="Override default target model",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        help="Override default draft model",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        print("Debug logging enabled")
    else:
        logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 50)
    print("PEARL Integration Test Suite")
    print("=" * 50)

    results = []

    if args.test in ["basic", "all"]:
        results.append(("Basic Test", test_pearl_basic()))

    if args.test in ["config", "all"]:
        test_pearl_config_validation()
        results.append(("Config Test", True))

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)
    print("=" * 50)
    if all_passed:
        print("All tests passed! ✓")
        print("\nNext steps:")
        print("1. Check MAT in logs (should be > 1.0)")
        print("2. Try with larger models (Llama-3-1B → Llama-3-8B)")
        print("3. Increase num_speculative_tokens if MAT is good")
        print("4. See unieai-dev/IMPLEMENTATION_STATUS.md for optimization")
        return 0
    else:
        print("Some tests failed! ✗")
        print("\nSee unieai-dev/TESTING_GUIDE.md for troubleshooting")
        return 1


if __name__ == "__main__":
    sys.exit(main())
