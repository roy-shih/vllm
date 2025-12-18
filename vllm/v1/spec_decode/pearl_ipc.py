import multiprocessing
import numpy as np
import torch
from multiprocessing import shared_memory
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class PearlIPCConfig:
    batch_size: int
    max_model_len: int
    gamma: int
    vocab_size: int
    shm_name_prefix: str = "pearl_ipc"

class PearlIPC:
    """
    Handles SharedMemory for PEARL Draft Worker.
    
    Structure:
    - Input Buffer: [batch_size, max_model_len] (int32) - Input Token IDs
    - Output Buffer: [batch_size, gamma] (int32) - Draft Token IDs
    - Metadata Buffer: [batch_size] (int32) - Current Sequence Lengths
    - Flags Buffer: [batch_size] (uint8) - Request Mode Flags
    
    Protocol Modes (Flags):
    - 0 (IPC_MODE_DELTA): Continuation. seqlens=WritePos, inputs[0]=NextToken.
    - 1 (IPC_MODE_NEW): New Request. seqlens=InputLen, inputs[:]=Context.
    - 2 (IPC_MODE_FEEDBACK): Verification Feedback. seqlens=VerifiedLen, inputs[0]=CorrectToken (if mismatch).
    """
    # IPC Protocol Constants
    IPC_MODE_DELTA = 0
    IPC_MODE_NEW = 1
    IPC_MODE_FEEDBACK = 2

    def __init__(self, config: PearlIPCConfig, create: bool = False):
        self.config = config
        self.create = create
        
        # Define shapes
        self.input_ids_shape = (config.batch_size, config.max_model_len)
        self.draft_shape = (config.batch_size, config.gamma)
        self.seq_lens_shape = (config.batch_size,)
        
        # Calculate sizes (int32 = 4 bytes)
        self.input_ids_size = int(np.prod(self.input_ids_shape)) * 4
        self.draft_size = int(np.prod(self.draft_shape)) * 4
        self.seq_lens_size = int(np.prod(self.seq_lens_shape)) * 4
        
        self.shm_name = config.shm_name_prefix
        
        if create:
            # unlink first just in case they exist from crash
            self._unlink_if_exists(self.shm_name)

        # 2. Allocate / Attached Shared Memory
        # Layout:
        # [Inputs (int32): batch * max_model_len]
        # output, seqlens, flags...
        
        input_size = self.config.batch_size * self.config.max_model_len * 4
        output_size = self.config.batch_size * self.config.gamma * 4
        seqlens_size = self.config.batch_size * 4
        flags_size = self.config.batch_size * 1 # uint8
        
        total_size = input_size + output_size + seqlens_size + flags_size
        
        if create:
            try:
                self.shm = shared_memory.SharedMemory(create=True, size=total_size, name=self.shm_name)
            except FileExistsError:
                self.shm = shared_memory.SharedMemory(create=False, name=self.shm_name)
        else:
             self.shm = shared_memory.SharedMemory(create=False, name=self.shm_name)

        # Create numpy views
        offset = 0
        self.np_inputs = np.ndarray((self.config.batch_size, self.config.max_model_len), dtype=np.int32, buffer=self.shm.buf, offset=offset)
        offset += input_size
        
        self.np_outputs = np.ndarray((self.config.batch_size, self.config.gamma), dtype=np.int32, buffer=self.shm.buf, offset=offset)
        offset += output_size
        
        self.np_seqlens = np.ndarray((self.config.batch_size,), dtype=np.int32, buffer=self.shm.buf, offset=offset)
        offset += seqlens_size

        self.np_is_new_req = np.ndarray((self.config.batch_size,), dtype=np.uint8, buffer=self.shm.buf, offset=offset)

    def _unlink_if_exists(self, name: str):
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def destroy(self):
        """Cleanup shared memory"""
        try:
            self.shm.close()
            if self.create:
                self.shm.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def get_input_tensor(self) -> torch.Tensor:
        """Returns tensor view of inputs (CPU)"""
        return torch.from_numpy(self.np_inputs)

    def get_output_tensor(self) -> torch.Tensor:
        """Returns tensor view of outputs (CPU)"""
        return torch.from_numpy(self.np_outputs)

    def get_seqlens_tensor(self) -> torch.Tensor:
        """Returns tensor view of sequence lengths (CPU)"""
        return torch.from_numpy(self.np_seqlens)
