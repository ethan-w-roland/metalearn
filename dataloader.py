"""
High-throughput loader that mmaps the .bin file and feeds CUDA batches with
async transfer + pinned memory.  Designed for single-GPU or DDP (each rank
gets its own worker pool).
"""

import numpy as np, torch, math, mmap, random, itertools, os
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

class BinDataset(IterableDataset):
    def __init__(self, bin_path: str, block_size: int):
        self.block_size = block_size
        self.n = os.path.getsize(bin_path) // 4   # uint32 tokens
        self.bin = open(bin_path, "rb")
        self.mm  = mmap.mmap(self.bin.fileno(), 0, access=mmap.ACCESS_READ)

    def _get(self, idx: int):
        start = idx * 4
        end   = start + 4 * self.block_size
        buf   = self.mm[start:end]
        return np.frombuffer(buf, dtype=np.uint32).copy()   # contiguous

    def __iter__(self):
        wi  = get_worker_info()
        g   = torch.Generator().manual_seed(1337 + (wi.id if wi else 0))
        while True:                         # infinite stream
            idx = torch.randint(0, self.n - self.block_size - 1, (1,), generator=g).item()
            yield self._get(idx)

def create_loader(bin_path, block_size, batch_size, num_workers=4, device="cuda"):
    ds = BinDataset(bin_path, block_size)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
        collate_fn=lambda xs: torch.as_tensor(xs, dtype=torch.long)  # [B, T]
    )
    # wrap to auto-transfer to GPU asynchronously
    for cpu_batch in dl:
        yield cpu_batch.to(device, non_blocking=True)

# Example usage:
# for batch in create_loader("./tiny_bin/train.bin", 2048, 8):
#     ...
