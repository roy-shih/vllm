
import unittest
import torch
import multiprocessing
import time
import os
import sys
from unittest.mock import MagicMock, patch

# Add repo root to path
sys.path.append(os.getcwd())

# MOCK DEPENDENCIES
from unittest.mock import MagicMock
sys.modules["cbor2"] = MagicMock()
sys.modules["vllm._version"] = MagicMock()
sys.modules["vllm.version"] = MagicMock()
# Mock version strings
sys.modules["vllm.version"].__version__ = "0.0.0"
sys.modules["vllm.version"].__version_tuple__ = (0, 0, 0)

from vllm.config import VllmConfig, SpeculativeConfig, ModelConfig, ParallelConfig
from vllm.v1.spec_decode.pearl_ipc import PearlIPCConfig
from vllm.v1.spec_decode.pearl_proposer import PEARLProposer

class TestPearlIPC(unittest.TestCase):
    def setUp(self):
        # Mock VLLM Config
        self.mock_vllm_config = MagicMock(spec=VllmConfig)
        self.mock_vllm_config.model_config = MagicMock(spec=ModelConfig)
        self.mock_vllm_config.model_config.max_model_len = 128
        self.mock_vllm_config.model_config.get_vocab_size.return_value = 100
        self.mock_vllm_config.model_config.dtype = torch.float16
        
        self.mock_vllm_config.scheduler_config = MagicMock()
        self.mock_vllm_config.scheduler_config.max_num_batched_tokens = 256
        self.mock_vllm_config.scheduler_config.max_num_seqs = 4

        self.mock_spec_config = MagicMock(spec=SpeculativeConfig)
        self.mock_spec_config.method = "pearl"
        self.mock_spec_config.num_speculative_tokens = 5
        self.mock_spec_config.pearl_gamma = 5
        self.mock_spec_config.pearl_draft_gpu_id = 0 # Use same GPU or CPU for test?
        # Note: Worker will try to set device. If no GPU, it might fail.
        # We need to mock torch.cuda in the worker if running on CPU-only env.
        
        self.mock_vllm_config.speculative_config = self.mock_spec_config
        self.mock_vllm_config.parallel_config = MagicMock(spec=ParallelConfig)

    @patch("vllm.v1.spec_decode.pearl_proposer.run_draft_worker_process")
    def test_proposer_spawn(self, mock_run_worker):
        # Test if proposer spawns process correctly
        device = torch.device("cpu")
        proposer = PEARLProposer(self.mock_vllm_config, device)
        proposer.load_model(None)
        
        self.assertIsNotNone(proposer.worker_process)
        self.assertTrue(proposer.worker_process.is_alive())
        
        # Cleanup
        proposer.shutdown()

    def test_ipc_communication_mock(self):
        # Testing logic without actually spawning a heavy process
        # We manually simulate the worker side in this test thread
        
        device = torch.device("cpu")
        proposer = PEARLProposer(self.mock_vllm_config, device)
        
        # Manually create the "Worker side" IPC
        worker_ipc = proposer.ipc # Uses same SHM since name is same
        
        # Create Dummy Inputs
        batch_size = 2
        next_token_ids = torch.tensor([10, 20], dtype=torch.int32)
        
        # Mock CommonAttentionMetadata
        mock_attn = MagicMock()
        mock_attn.query_start_loc = torch.tensor([0, 5, 10], dtype=torch.int32)
        
        # Mock target_token_ids
        target_ids = torch.randint(0, 100, (10,), dtype=torch.int32)
        
        # Start Proposer in a separate thread/process? 
        # Actually Proposer.propose() blocks waiting for event.
        # So we need to simulate the "Worker" in a thread that waits for start_event.
        
        import threading
        def worker_simulation():
            print("Worker simulator waiting...")
            if proposer.start_event.wait(timeout=2):
                print("Worker received signal!")
                # Read Inputs
                # Check inputs
                inputs = proposer.ipc.np_inputs
                seqlens = proposer.ipc.np_seqlens
                
                print(f"Worker read seqlens: {seqlens[0]}, {seqlens[1]}")
                
                # Write Outputs
                proposer.ipc.np_outputs[:2, :5] = 99 # Fill with 99
                
                proposer.done_event.set()
        
        t = threading.Thread(target=worker_simulation)
        t.start()
        
        # Call Propose
        print("Proposer calling propose...")
        output = proposer.propose(
            target_token_ids=target_ids,
            target_positions=None,
            target_hidden_states=None,
            next_token_ids=next_token_ids,
            last_token_indices=None,
            common_attn_metadata=mock_attn,
            sampling_metadata=None
        )
        
        t.join()
        
        print("Proposer returned:", output)
        
        # Assertions
        self.assertEqual(output.shape, (batch_size, 5))
        self.assertEqual(output[0, 0].item(), 99)
        
        proposer.shutdown()

if __name__ == "__main__":
    unittest.main()
