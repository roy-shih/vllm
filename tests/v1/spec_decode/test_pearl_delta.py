import torch
import unittest
from unittest.mock import MagicMock
import numpy as np
import sys
import os

# Mock vLLM imports
sys.modules["vllm.config"] = MagicMock()
sys.modules["vllm.logger"] = MagicMock()
sys.modules["vllm.v1.attention.backends.utils"] = MagicMock()
sys.modules["vllm.v1.sample.metadata"] = MagicMock()
sys.modules["vllm.v1.spec_decode.pearl_worker"] = MagicMock()

# Import Proposer after mocking
# We need to ensure we can import the file. 
# Assuming script runs from root of repo.
sys.path.append(os.getcwd())

from vllm.v1.spec_decode.pearl_proposer import PEARLProposer
from vllm.v1.spec_decode.pearl_ipc import PearlIPCConfig

class MockCommonAttentionMetadata:
    def __init__(self, start_loc_list):
        self.query_start_loc = torch.tensor(start_loc_list, dtype=torch.int32)

class TestPearlDeltaProtocol(unittest.TestCase):
    def setUp(self):
        # Mock Config
        self.vllm_config = MagicMock()
        self.vllm_config.speculative_config.method = "pearl"
        self.vllm_config.speculative_config.num_speculative_tokens = 5
        self.vllm_config.speculative_config.pearl_gamma = 5
        self.vllm_config.speculative_config.pearl_draft_gpu_id = 0
        self.vllm_config.model_config.max_model_len = 128
        self.vllm_config.scheduler_config.max_num_batched_tokens = 128
        self.vllm_config.scheduler_config.max_num_seqs = 2
        self.vllm_config.model_config.get_vocab_size.return_value = 1000

        self.device = torch.device("cpu")
        
        # Instantiate Proposer (mocking runner)
        self.mock_runner = MagicMock()
        # Initial State: Empty Batch
        self.mock_runner.input_batch.req_ids = []
        
        self.proposer = PEARLProposer(self.vllm_config, self.device, runner=self.mock_runner)
        # Mock Worker Process as alive
        self.proposer.worker_process = MagicMock()
        self.proposer.worker_process.is_alive.return_value = True
        
        # Mock IPC output (so wait returns)
        # We simulate worker writing output immediately? No, we just need propose to finish.
        # But propose waits for done_event.
        # We need to spawn a thread to set done_event or mock done_event.wait.
        self.proposer.done_event.wait = MagicMock(return_value=True) # Immediate success
        
        # Zero out output buffer for cleanliness
        self.proposer.ipc.np_outputs[:] = 0

    def tearDown(self):
        self.proposer.shutdown()

    def test_delta_logic(self):
        """
        Verify:
        1. Step 1: New Req -> IsNew=1, Full Context Copied.
        2. Step 2: Same Req -> IsNew=0, Only New Token Copied (at pos 0), SeqLen=StartPos.
        3. Step 3: Different Req -> IsNew=1, Full Context Copied.
        """
        
        # --- Step 1: New Request "ReqA" ---
        self.mock_runner.input_batch.req_ids = ["ReqA", "ReqB"]
        
        # Inputs: Context Len 3. [1, 2, 3]
        target_token_ids = torch.tensor([1, 2, 3, 10, 20, 30], dtype=torch.int32) # Flattened batch of 2
        # Batch 0: [1, 2, 3] (len 3)
        # Batch 1: [10, 20, 30] (len 3)
        common_meta = MockCommonAttentionMetadata([0, 3, 6]) 
        
        next_token_ids = torch.tensor([4, 40], dtype=torch.int32)
        
        self.proposer.propose(
            target_token_ids=target_token_ids,
            target_positions=None, target_hidden_states=None,
            next_token_ids=next_token_ids, last_token_indices=None,
            common_attn_metadata=common_meta, sampling_metadata=None
        )
        
        # Analyze IPC Buffers
        ipc_flags = self.proposer.ipc.np_is_new_req
        ipc_inputs = self.proposer.ipc.np_inputs
        ipc_seqlens = self.proposer.ipc.np_seqlens
        
        print(f"Step 1 Flags: {ipc_flags[:2]}")
        
        # Assertions
        self.assertEqual(ipc_flags[0], 1, "Slot 0 should be New")
        self.assertEqual(ipc_flags[1], 1, "Slot 1 should be New")
        
        # Check Inputs Content: [1, 2, 3, 4]
        self.assertTrue(np.array_equal(ipc_inputs[0, :4], [1, 2, 3, 4]))
        self.assertEqual(ipc_seqlens[0], 4, "Len should be 4 (3+1)")

        # --- Step 2: Continue "ReqA", Continue "ReqB" ---
        # Context grows to 4 (History now implies [1, 2, 3, 4])
        # Next tokens: 5, 50.
        target_token_ids = torch.tensor([1, 2, 3, 4, 10, 20, 30, 40], dtype=torch.int32)
        common_meta = MockCommonAttentionMetadata([0, 4, 8])
        next_token_ids = torch.tensor([5, 50], dtype=torch.int32)
        
        self.proposer.propose(
            target_token_ids=target_token_ids,
            target_positions=None, target_hidden_states=None,
            next_token_ids=next_token_ids, last_token_indices=None,
            common_attn_metadata=common_meta, sampling_metadata=None
        )
        
        print(f"Step 2 Flags: {ipc_flags[:2]}")
        print(f"Step 2 Seqlens: {ipc_seqlens[:2]}")
        print(f"Step 2 Inputs[0,0]: {ipc_inputs[0, 0]}")

        # Assertions
        self.assertEqual(ipc_flags[0], 0, "Slot 0 should be Delta")
        self.assertEqual(ipc_flags[1], 0, "Slot 1 should be Delta")
        
        # Check Delta Payload
        self.assertEqual(ipc_inputs[0, 0], 5, "Should carry NEW token (5) at index 0")
        
        # Check Start Position (Overloaded SeqLen)
        # Previous history was 4. Start Position for new token is 4.
        self.assertEqual(ipc_seqlens[0], 4, "Write Pos should be 4")
        
        # --- Step 3: Replace "ReqB" with "ReqC" (Slot 1 Change) ---
        self.mock_runner.input_batch.req_ids = ["ReqA", "ReqC"]
        
        # ReqA: continues (History 5). Next=6.
        # ReqC: New (History 2: [100, 200]). Next=300.
        
        target_token_ids = torch.tensor([1, 2, 3, 4, 5, 100, 200], dtype=torch.int32)
        common_meta = MockCommonAttentionMetadata([0, 5, 7])
        next_token_ids = torch.tensor([6, 300], dtype=torch.int32)
        
        self.proposer.propose(
            target_token_ids=target_token_ids,
            target_positions=None, target_hidden_states=None,
            next_token_ids=next_token_ids, last_token_indices=None,
            common_attn_metadata=common_meta, sampling_metadata=None
        )
        
        print(f"Step 3 Flags: {ipc_flags[:2]}")
        
        # Assertions
        self.assertEqual(ipc_flags[0], 0, "Slot 0 should still be Delta (ReqA)")
        self.assertEqual(ipc_flags[1], 1, "Slot 1 should be New (ReqC)")
        
        # Check Content
        self.assertEqual(ipc_inputs[0, 0], 6, "Slot 0 Delta Token")
        self.assertEqual(ipc_seqlens[0], 5, "Slot 0 Write Pos 5")
        
        self.assertTrue(np.array_equal(ipc_inputs[1, :3], [100, 200, 300]))
        self.assertEqual(ipc_seqlens[1], 3, "Slot 1 Full Len 3")

    pass

if __name__ == "__main__":
    unittest.main()
