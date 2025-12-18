
import unittest
import torch
import numpy as np
import os
import sys

# Add repo root to path
sys.path.append(os.getcwd())

from vllm.v1.spec_decode.pearl_ipc import PearlIPC, PearlIPCConfig

class TestPearlIPCIsolated(unittest.TestCase):
    def test_ipc_creation_and_tensor_io(self):
        config = PearlIPCConfig(
            batch_size=4,
            max_model_len=128,
            gamma=5,
            vocab_size=1000,
            shm_name_prefix="test_ipc_iso"
        )
        
        # Creator
        ipc_owner = PearlIPC(config, create=True)
        
        # Reader (simulating other process)
        ipc_reader = PearlIPC(config, create=False)
        
        # Test 1: Write to Inputs via Owner
        input_tensor_owner = ipc_owner.get_input_tensor() # torch wrapper
        input_tensor_owner.fill_(0)
        input_tensor_owner[0, :5] = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        
        # Read via Reader
        input_tensor_reader = ipc_reader.get_input_tensor()
        self.assertTrue(torch.equal(input_tensor_reader[0, :5], torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)))
        
        # Test 2: Write to Outputs via Reader (Worker)
        output_tensor_reader = ipc_reader.get_output_tensor()
        output_tensor_reader.fill_(0)
        output_tensor_reader[0, :] = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32)
        
        # Read via Owner
        output_tensor_owner = ipc_owner.get_output_tensor()
        self.assertTrue(torch.equal(output_tensor_owner[0, :], torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32)))
        
        # Cleanup
        ipc_reader.destroy() # Close
        ipc_owner.destroy() # Unlink
        
if __name__ == "__main__":
    unittest.main()
