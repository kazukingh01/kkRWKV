import torch
import numpy as np


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, n_features: int, n_symbols: int, n_label: int=5, seq_len: int=128, n_samples: int=1000000):
        self.n_features = n_features
        self.n_symbols  = n_symbols
        self.seq_len    = seq_len
        self.data       = np.random.rand(n_samples, n_features).astype(np.float32)
        self.gt         = np.random.randint(0, n_label, (n_samples, n_symbols)).astype(np.int64)
        self.indexes    = np.arange(seq_len, len(self.data), seq_len // 4)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx: int):
        _idx = self.indexes[idx]
        return torch.tensor(self.data[_idx - self.seq_len:_idx]), torch.tensor(self.gt[_idx])
