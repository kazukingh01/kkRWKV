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

class RandomDataset2(torch.utils.data.Dataset):
    def __init__(self, n_feat: int, n_label_1: int, n_label_2: int, n_feat_other: int, n_symbols: int, n_label: int=5, seq_len: int=128, n_samples: int=1000000):
        self.n_feat       = n_feat
        self.n_label_1    = n_label_1
        self.n_label_2    = n_label_2
        self.n_feat_other = n_feat_other
        self.seq_len      = seq_len
        self.data_p       = torch.from_numpy(np.random.rand(n_samples, n_feat      ).astype(np.float32))
        self.data_l1      = torch.from_numpy(np.random.randint(0, n_label_1, (n_samples, )).astype(np.int64))
        self.data_l2      = torch.from_numpy(np.random.randint(0, n_label_2, (n_samples, )).astype(np.int64))
        self.data_oth     = torch.from_numpy(np.random.rand(n_samples, n_feat_other).astype(np.float32))
        self.gt           = torch.from_numpy(np.random.randint(0, n_label, (n_samples, n_symbols)).astype(np.int64))
        self.indexes      = np.arange(seq_len, len(self.data_p), seq_len // 4)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx: int):
        _idx = self.indexes[idx]
        _idx_fr = _idx + 1 - self.seq_len
        _idx_to = _idx + 1
        data    = torch.cat([
            self.data_p[_idx_fr:_idx_to],
            self.data_l1[_idx_fr:_idx_to].reshape(-1, 1),
            self.data_l2[_idx_fr:_idx_to].reshape(-1, 1),
            self.data_oth[_idx_fr:_idx_to]
        ], dim=-1)
        return data, self.gt[_idx]
