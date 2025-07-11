import glob
import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch.utils.data.distributed import DistributedSampler


class TokenDataset(Dataset):
    """PyTorch Dataset for memory-mapped token files with optional shuffling."""
    
    def __init__(
        self,
        filename: str,
        B: int,
        T: int,
        process_rank: int = 0,
        num_processes: int = 1,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.filename = filename
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.shuffle = shuffle
        self.seed = seed
        
        # Load tokens as memory map
        self.tokens = np.memmap(filename, dtype=np.uint16, mode='r')
        self.num_tokens = len(self.tokens)
        
        # Calculate number of sequences that can be formed
        self.sequence_length = B * T + 1  # +1 for target
        self.num_sequences = (self.num_tokens - 1) // self.sequence_length
        
        # For distributed training, each process gets a subset
        self.sequences_per_process = self.num_sequences // self.num_processes
        self.start_idx = self.process_rank * self.sequences_per_process
        
        # Create shuffled indices if needed
        if self.shuffle:
            self.rng = np.random.RandomState(seed)
            self.indices = self.rng.permutation(self.num_sequences)
        else:
            self.indices = np.arange(self.num_sequences)
    
    def __len__(self):
        return self.sequences_per_process
    
    def __getitem__(self, idx):
        # Map local index to global index
        global_idx = self.start_idx + idx
        
        # Use shuffled index if shuffling
        if self.shuffle:
            global_idx = self.indices[global_idx]
        
        # Calculate token position
        start_pos = global_idx * self.sequence_length
        
        # Handle edge case where we might go beyond file
        if start_pos + self.sequence_length > self.num_tokens:
            # Wrap around to beginning
            indices = np.arange(start_pos, start_pos + self.sequence_length) % self.num_tokens
            buf = self.tokens[indices]
        else:
            buf = self.tokens[start_pos:start_pos + self.sequence_length]
        
        # Convert to tensor
        buf = torch.from_numpy(buf.astype(np.int64))
        
        # Split into input and target
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        
        return x, y


class ShardedTokenDataset(Dataset):
    """Dataset that handles multiple sharded files."""
    
    def __init__(
        self,
        filename_pattern: str,
        B: int,
        T: int,
        process_rank: int = 0,
        num_processes: int = 1,*
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.shuffle = shuffle
        self.seed = seed
        self.sequence_length = B * T + 1
        
        # Find all files matching pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"No files found matching pattern: {filename_pattern}"
        
        # Load file sizes and compute indices
        self.file_sizes = []
        self.file_offsets = [0]
        self.num_tokens = 0
        
        for fname in self.files:
            tokens = np.memmap(fname, dtype=np.uint16, mode='r')
            size = len(tokens)
            self.file_sizes.append(size)
            self.num_tokens += size
            self.file_offsets.append(self.num_tokens)
        
        self.num_sequences = (self.num_tokens - 1) // self.sequence_length
        
        # Distributed setup
        self.sequences_per_process = self.num_sequences // self.num_processes
        self.start_idx = self.process_rank * self.sequences_per_process
        
        # Shuffling
        if self.shuffle:
            self.rng = np.random.RandomState(seed)
            self.indices = self.rng.permutation(self.num_sequences)
        else:
            self.indices = np.arange(self.num_sequences)
        
        # Cache for memory maps
        self.mmap_cache = {}
    
    def _get_mmap(self, file_idx):
        """Get memory map for file, with caching."""
        if file_idx not in self.mmap_cache:
            self.mmap_cache[file_idx] = np.memmap(
                self.files[file_idx], dtype=np.uint16, mode='r'
            )
        return self.mmap_cache[file_idx]
    
    def _get_tokens(self, global_pos, length):
        """Get tokens from appropriate file(s)."""
        # Find which file contains this position
        file_idx = 0
        for i in range(len(self.files)):
            if global_pos < self.file_offsets[i + 1]:
                file_idx = i
                break
        
        local_pos = global_pos - self.file_offsets[file_idx]
        mmap = self._get_mmap(file_idx)
        
        # Check if we need to span multiple files
        if local_pos + length <= len(mmap):
            # Simple case: all in one file
            return mmap[local_pos:local_pos + length]
        else:
            # Complex case: spans multiple files
            result = []
            remaining = length
            current_file = file_idx
            current_pos = local_pos
            
            while remaining > 0:
                mmap = self._get_mmap(current_file)
                available = len(mmap) - current_pos
                to_read = min(remaining, available)
                
                result.append(mmap[current_pos:current_pos + to_read])
                
                remaining -= to_read
                current_file += 1
                current_pos = 0
                
                # Wrap around if we run out of files
                if current_file >= len(self.files):
                    current_file = 0
            
            return np.concatenate(result)
    
    def __len__(self):
        return self.sequences_per_process
    
    def __getitem__(self, idx):
        # Map to global index
        global_idx = self.start_idx + idx
        
        # Use shuffled index if needed
        if self.shuffle:
            global_idx = self.indices[global_idx]
        
        # Calculate global token position
        global_pos = global_idx * self.sequence_length
        
        # Get tokens
        buf = self._get_tokens(global_pos, self.sequence_length)
        buf = torch.from_numpy(buf.astype(np.int64))
        
        # Split into input and target
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        
        return x, y


class SingleDataLoader:
    """Improved DataLoader using PyTorch's built-in functionality."""
    
    def __init__(
        self,
        filename_pattern: str,
        B: int,
        T: int,
        process_rank: int,
        num_processes: int,
        label: str,
        num_workers: int = 4,
        shuffle: bool = False,
        seed: int = 42,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        
        self.B = B
        self.T = T
        self.label = label
        self.device = device
        
        # Check if pattern matches multiple files
        files = glob.glob(filename_pattern)
        
        if len(files) == 1:
            # Single file dataset
            self.dataset = TokenDataset(
                files[0], B, T, process_rank, num_processes, shuffle, seed
            )
        else:
            # Multiple files dataset
            self.dataset = ShardedTokenDataset(
                filename_pattern, B, T, process_rank, num_processes, shuffle, seed
            )
        
        # Create PyTorch DataLoader
        self.dataloader = TorchDataLoader(
            self.dataset,
            batch_size=1,  # Dataset already handles batching
            shuffle=False,  # Shuffling handled in dataset
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=False,
        )
        
        # Create iterator
        self.iterator = iter(self.dataloader)
        
        # Track total tokens for compatibility
        self.num_tokens = self.dataset.num_tokens
    
    def reset(self):
        """Reset the dataloader iterator."""
        self.iterator = iter(self.dataloader)
    
    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get next batch, with automatic reset on StopIteration."""
        try:
            x, y = next(self.iterator)
            # Remove batch dimension added by DataLoader
            x = x.squeeze(0)
            y = y.squeeze(0)
            
            # Move to device if specified
            if self.device is not None:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
            
            return x, y, self.label
        except StopIteration:
            self.reset()
            return self.next_batch()
    
    def __len__(self):
        return len(self.dataset)


class InterleavedDataLoader:
    """Randomly interleave multiple DataLoaders with improved performance."""
    
    def __init__(
        self, 
        loaders: Sequence["SingleDataLoader"],
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        
        assert len(loaders) >= 2, "InterleavedDataLoader requires at least two component loaders."
        self.loaders: List["SingleDataLoader"] = list(loaders)
        self.device = device
        
        # Validate that all loaders share the same batch and sequence sizes
        base_B, base_T = self.loaders[0].B, self.loaders[0].T
        for ld in self.loaders[1:]:
            assert ld.B == base_B and ld.T == base_T, "All loaders must have the same B and T values."
        self.B = base_B
        self.T = base_T
        
        # Aggregate total token count
        self.ntok_total: int = sum(ld.ntok_total for ld in self.loaders)
        
        # Internal state for randomized cycling
        self._queue: List[int] = []
        self._last_idx: Optional[int] = None
        self.reset()
    
    def _refill_queue(self) -> None:
        """Generate a new random permutation of loader indices."""
        indices = list(range(len(self.loaders)))
        random.shuffle(indices)
        
        # Ensure no immediate repetition
        if self._last_idx is not None and indices[0] == self._last_idx and len(indices) > 1:
            indices[0], indices[1] = indices[1], indices[0]
        
        self._queue = indices
    
    def reset(self) -> None:
        """Reset all component loaders and internal permutation state."""
        for ld in self.loaders:
            ld.reset()
        self._last_idx = None
        self._queue = []
        self._refill_queue()
    
    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Return the next batch according to the random interleaving schedule."""
        if not self._queue:
            self._refill_queue()
        
        idx = self._queue.pop(0)
        x, y, label = self.loaders[idx].next_batch()
        self._last_idx = idx
        
        # Ensure device consistency
        if self.device is not None:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        
        return x, y, label
    
    def __len__(self):
        return sum(len(ld) for ld in self.loaders)