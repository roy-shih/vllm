
import unittest
import torch
import multiprocessing
import numpy as np
import time
from unittest.mock import MagicMock, patch
import sys

import sys
import os

import sys
import os
from unittest.mock import MagicMock

import sys
import os
from unittest.mock import MagicMock

import types

# Ensure project root is in path
sys.path.append(os.getcwd())

# Mock dependencies that might be missing in this environment
sys.modules["cbor2"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["vllm._version"] = MagicMock()
sys.modules["vllm.env_override"] = MagicMock()

# Mock vllm.utils as a package
utils_pkg = types.ModuleType("vllm.utils")
sys.modules["vllm.utils"] = utils_pkg
sys.modules["vllm.utils.torch_utils"] = MagicMock()
sys.modules["vllm.utils.math_utils"] = MagicMock()
sys.modules["vllm.utils.network_utils"] = MagicMock() # catch usage

# Mock vllm.v1.attention.backends.utils to stop import chain
# Do NOT mock vllm.v1 itself, as we need to import spec_decode from it
sys.modules["vllm.v1.attention"] = MagicMock()
sys.modules["vllm.v1.attention.backends"] = MagicMock()
sys.modules["vllm.v1.attention.backends.utils"] = MagicMock()

# Mock Sampling Metadata
sys.modules["vllm.v1.sample"] = MagicMock()
sys.modules["vllm.v1.sample.metadata"] = MagicMock()
sys.modules["vllm.logits_process"] = MagicMock()
sys.modules["vllm.tokenizers"] = MagicMock()

sys.modules["vllm.config"] = MagicMock()
sys.modules["vllm.distributed"] = MagicMock() # Kill distributed imports
sys.modules["vllm.inputs"] = MagicMock() 
sys.modules["vllm.logger"] = MagicMock()

# Mock model_executor as package or just mock the worker?
# Mocking worker is safer to avoid deeper imports
worker_mock = MagicMock()
sys.modules["vllm.v1.spec_decode.pearl_worker"] = worker_mock
sys.modules["vllm.model_executor"] = MagicMock() # Fallback

# Import Proposer and IPC
from vllm.v1.spec_decode.pearl_ipc import PearlIPC, PearlIPCConfig
from vllm.v1.spec_decode.pearl_proposer import PEARLProposer
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

class TestPearlAsync(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.max_len = 32
        self.gamma = 2
        
        self.ipc_config = PearlIPCConfig(
            batch_size=self.batch_size,
            max_model_len=self.max_len,
            gamma=self.gamma,
            vocab_size=100
        )
        
        # Setup IPC
        self.ipc = PearlIPC(self.ipc_config, create=True)
        
        # Mocks
        self.runner_mock = MagicMock()
        self.runner_mock.input_batch.req_ids = ["req1", "req2"]
        
        self.vllm_config_mock = MagicMock()
        # Mocking config attributes required by PEARLProposer
        self.vllm_config_mock.speculative_config.method = "pearl"
        self.vllm_config_mock.speculative_config.num_speculative_tokens = self.gamma
        self.vllm_config_mock.speculative_config.pearl_gamma = self.gamma
        self.vllm_config_mock.speculative_config.pearl_draft_gpu_id = 0
        self.vllm_config_mock.model_config.max_model_len = self.max_len
        self.vllm_config_mock.scheduler_config.max_num_batched_tokens = 100
        self.vllm_config_mock.scheduler_config.max_num_seqs = self.batch_size
        self.vllm_config_mock.speculative_config.pearl_ipc_buffer_size = 1024*1024 # Dummy
        
        # Init Proposer with IPC injection
        # We need to hack __init__ slightly or pass config that avoids real spawning
        
        with patch("vllm.v1.spec_decode.pearl_proposer.multiprocessing.Process"):
             self.proposer = PEARLProposer(
                 vllm_config=self.vllm_config_mock,
                 device="cpu", # Test on CPU
                 runner=self.runner_mock
             )
        
        # Capture the IPC created by proposer
        self.ipc = self.proposer.ipc
        
        # Inject our runner and state tweaks
        self.proposer.last_req_ids = [None] * self.proposer.batch_size_limit
        self.proposer.read_ahead_enabled = True # Testing Async
        self.proposer.current_draft_tokens = self.gamma
        
        # Events (Mocked worker implies events might need manual set if not spawned)
        # PEARLProposer init creates events. We use them.
        self.proposer.worker_process = MagicMock()
        self.proposer.worker_process.is_alive.return_value = True

    def tearDown(self):
        if hasattr(self, 'ipc'):
            self.ipc.destroy()

    def test_cold_start(self):
        """Verify Cold Start (New Request) behavior - Expect WAIT"""
        print("\n[Test] Cold Start...")
        
        # Mock Inputs (FLATTENED for Continuous Batching)
        # 2 requests, len 10 each. Total 20 tokens.
        target_tokens = torch.zeros((20,), dtype=torch.int32) 
        next_tokens = torch.tensor([101, 102], dtype=torch.int32)
        
        # Metadata logic required: query_start_loc
        common_meta = MagicMock()
        common_meta.query_start_loc = torch.tensor([0, 10, 20], dtype=torch.int32)
        
        # Simulate WORKER running in background
        def worker_sim():
            while not self.proposer.start_event.is_set():
                time.sleep(0.01)
            # Received signal
            self.proposer.start_event.clear()
            # Verify Flags (Should be NEW=1)
            flags = self.ipc.np_is_new_req[:self.batch_size]
            assert np.all(flags == 1), f"Expected New Req flags 1, got {flags}"
            
            # Write Output
            self.ipc.np_outputs[:, :] = 999
            self.proposer.done_event.set()

        import threading
        t = threading.Thread(target=worker_sim)
        t.start()
        
        # Run Propose
        output = self.proposer.propose(
            target_token_ids=target_tokens,
            target_positions=None,
            target_hidden_states=None,
            next_token_ids=next_tokens,
            last_token_indices=None,
            common_attn_metadata=common_meta,
            sampling_metadata=None
        )
        
        t.join()
        
        # Verify Output
        self.assertTrue((output == 999).all())
        # Verify internal state: last_expected_pos should be updated
        # New Req writes at TotalLen (11). Next expected = 11 + gamma = 13.
        expected = [13, 13] 
        self.assertTrue(np.all(self.proposer.last_expected_pos[:2] == expected))
        print("[Pass] Cold Start")

    def test_speculation_hit(self):
        """Verify Read-Ahead HIT (Zero Latency)"""
        print("\n[Test] Speculation HIT...")
        
        # 1. Setup Pre-conditions (Simulate previous turn finished)
        self.proposer.last_req_ids = ["req1", "req2"]
        # Expected position set to what we are about to request
        # If current history is 20, Next Write is 20.
        # So last_expected_pos must be 20 for HIT.
        self.proposer.last_expected_pos[:2] = 20
        
        # 2. Simulate Worker has ALREADY finished speculation for Pos 20
        self.ipc.np_outputs[:, :] = 777 # Speculative Result
        self.proposer.done_event.set() # Worker is DONE ahead of time
        
        # 3. Inputs for this turn (Delta)
        # Context Len = 20.
        # Next Token = [201, 202]
        target_tokens_dummy = torch.zeros((self.batch_size,), dtype=torch.int32) # Dummy (Flat)
        next_tokens = torch.tensor([201, 202], dtype=torch.int32)
        common_meta = MagicMock()
        # Context length logic: start_loc=[0, 20, 40]. len=20.
        common_meta.query_start_loc = torch.tensor([0, 20, 40], dtype=torch.int32)

        # 4. Run Propose
        start_time = time.time()
        output = self.proposer.propose(
            target_token_ids=target_tokens_dummy, # Must be tensor
            target_positions=None,
            target_hidden_states=None,
            next_token_ids=next_tokens,
            last_token_indices=None,
            common_attn_metadata=common_meta,
            sampling_metadata=None
        )
        duration = time.time() - start_time
        
        # 5. Verification
        # Should return 777 immediately
        self.assertTrue((output == 777).all())
        # Should trigger start_event (ACK) but NOT clear/wait loop
        self.assertTrue(self.proposer.start_event.is_set())
        
        # Verify internal state: last_expected_pos updated
        # Current Write = 20. Next Expected = 22.
        expected = [22, 22]
        self.assertTrue(np.all(self.proposer.last_expected_pos[:2] == expected))
        
        # Verify IPC Flags: Should be 0 (Delta)
        flags = self.ipc.np_is_new_req[:self.batch_size]
        self.assertTrue(np.all(flags == 0))
        
        print(f"[Pass] Speculation HIT (Duration: {duration:.5f}s)")

    def test_speculation_miss(self):
        """Verify Read-Ahead MISS (Rewind/Wait)"""
        print("\n[Test] Speculation MISS...")
        
        # 1. Setup Pre-conditions
        self.proposer.last_req_ids = ["req1", "req2"]
        # Expecting Pos 20, bu
        self.proposer.last_expected_pos[:2] = 20
        
        # 2. Simulate Worker has finished speculation but for WRONG prediction
        # e.g. Worker thought next was 20, but actually Rewind happened to 18.
        self.ipc.np_outputs[:, :] = 888 # Garbage
        self.proposer.done_event.set()
        
        # 3. Inputs (Rewind Scenario)
        # Context Len = 18 (Last accepted was at 17).
        target_tokens_dummy = torch.zeros((self.batch_size,), dtype=torch.int32)
        next_tokens = torch.tensor([301, 302], dtype=torch.int32)
        common_meta = MagicMock()
        common_meta.query_start_loc = torch.tensor([0, 18, 36], dtype=torch.int32)
        
        # 4. Worker SIM (Must react to Rewind)
        def worker_sim():
            # Wait for Proposer to signal Rewind
            while not self.proposer.start_event.is_set():
                time.sleep(0.01)
            
            # Verify IPC SEQLENS (Should be 18 - Write Pos)
            # And Flags should be 0 (Delta)
            # The MISMATCH is logical (Write 18 != Expected 20)
            
            seqlens = self.ipc.np_seqlens[:2]
            assert np.all(seqlens == 18), f"Expected rewind len 18, got {seqlens}"
            
            # Write Corrected Output
            self.ipc.np_outputs[:, :] = 555
            self.proposer.done_event.set()

        import threading
        t = threading.Thread(target=worker_sim)
        t.start()
        
        # 5. Run Propose
        output = self.proposer.propose(
            target_token_ids=target_tokens_dummy,
            target_positions=None,
            target_hidden_states=None,
            next_token_ids=next_tokens,
            last_token_indices=None,
            common_attn_metadata=common_meta,
            sampling_metadata=None
        )
        
        t.join()
        
        # 6. Verification
        self.assertTrue((output == 555).all())
        # Internal state updated
        # Current Write = 18. Next Expected = 20.
        expected = [20, 20]
        self.assertTrue(np.all(self.proposer.last_expected_pos[:2] == expected))
        print("[Pass] Speculation MISS (Rewind)")

if __name__ == "__main__":
    unittest.main()
