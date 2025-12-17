#!/usr/bin/env python3
"""
Simple structure test for PEARL integration.
Tests basic code structure without requiring model loading.
"""

import sys
import ast

def test_pearl_proposer_structure():
    """Test that PEARLProposer has the required methods."""
    print("Testing PEARL proposer structure...")

    # Parse the file
    with open("vllm/v1/spec_decode/pearl_proposer.py", "r") as f:
        tree = ast.parse(f.read())

    # Find the PEARLProposer class
    pearl_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PEARLProposer":
            pearl_class = node
            break

    if pearl_class is None:
        print("✗ PEARLProposer class not found")
        return False

    print("✓ PEARLProposer class found")

    # Check for required methods
    required_methods = [
        "__init__",
        "load_model",
        "propose",
        "_generate_draft_tokens_for_sequence",
        "log_stats",
    ]

    found_methods = set()
    for node in pearl_class.body:
        if isinstance(node, ast.FunctionDef):
            found_methods.add(node.name)

    print(f"\nFound methods: {sorted(found_methods)}")

    missing = set(required_methods) - found_methods
    if missing:
        print(f"✗ Missing methods: {missing}")
        return False

    print(f"✓ All required methods present")

    # Check propose method signature
    propose_method = None
    for node in pearl_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == "propose":
            propose_method = node
            break

    if propose_method:
        # Get argument names
        args = [arg.arg for arg in propose_method.args.args]
        print(f"\npropose() arguments: {args}")

        required_args = [
            "self",
            "target_token_ids",
            "target_positions",
            "target_hidden_states",
            "next_token_ids",
            "last_token_indices",
            "common_attn_metadata",
            "sampling_metadata",
            "mm_embed_inputs",
        ]

        if set(required_args).issubset(set(args)):
            print("✓ propose() has EAGLE-style signature")
        else:
            missing_args = set(required_args) - set(args)
            print(f"✗ propose() missing arguments: {missing_args}")
            return False

    return True


def test_gpu_model_runner_integration():
    """Test that gpu_model_runner.py has PEARL integration."""
    print("\n" + "="*50)
    print("Testing gpu_model_runner integration...")

    with open("vllm/v1/worker/gpu_model_runner.py", "r") as f:
        content = f.read()

    # Check for PEARL import
    if "from vllm.v1.spec_decode.pearl_proposer import PEARLProposer" in content:
        print("✓ PEARLProposer imported")
    else:
        print("✗ PEARLProposer not imported")
        return False

    # Check for PEARL initialization
    if 'self.speculative_config.method == "pearl"' in content:
        print("✓ PEARL method check present")
    else:
        print("✗ PEARL method check not found")
        return False

    if "PEARLProposer(self.vllm_config, self.device, self)" in content:
        print("✓ PEARL initialization code present")
    else:
        print("✗ PEARL initialization code not found")
        return False

    # Check for propose call
    if "self.drafter.propose(" in content:
        print("✓ propose() call present")
    else:
        print("✗ propose() call not found")
        return False

    return True


def test_speculative_config():
    """Test that speculative config has PEARL parameters."""
    print("\n" + "="*50)
    print("Testing speculative config...")

    with open("vllm/config/speculative.py", "r") as f:
        content = f.read()

    # Check for pearl in SpeculativeMethod
    if '"pearl"' in content or "'pearl'" in content:
        print("✓ 'pearl' in SpeculativeMethod")
    else:
        print("✗ 'pearl' not found in SpeculativeMethod")
        return False

    # Check for PEARL config parameters
    pearl_params = [
        "pearl_gamma",
        "pearl_max_num_batched_tokens",
        "pearl_max_num_seqs",
    ]

    for param in pearl_params:
        if param in content:
            print(f"✓ {param} parameter present")
        else:
            print(f"✗ {param} parameter not found")
            return False

    return True


def main():
    print("="*50)
    print("PEARL Integration Structure Test")
    print("="*50)

    results = []

    # Test 1: PEARLProposer structure
    try:
        results.append(("PEARLProposer structure", test_pearl_proposer_structure()))
    except Exception as e:
        print(f"✗ Error testing PEARLProposer: {e}")
        results.append(("PEARLProposer structure", False))

    # Test 2: gpu_model_runner integration
    try:
        results.append(("gpu_model_runner integration", test_gpu_model_runner_integration()))
    except Exception as e:
        print(f"✗ Error testing gpu_model_runner: {e}")
        results.append(("gpu_model_runner integration", False))

    # Test 3: Speculative config
    try:
        results.append(("Speculative config", test_speculative_config()))
    except Exception as e:
        print(f"✗ Error testing speculative config: {e}")
        results.append(("Speculative config", False))

    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)
    print("="*50)
    if all_passed:
        print("All structure tests passed! ✓")
        print("\nThe code structure is correct.")
        print("Next: Test with actual models using test_pearl_integration.py")
        return 0
    else:
        print("Some tests failed! ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
