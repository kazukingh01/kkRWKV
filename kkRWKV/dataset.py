import torch
import pandas as pd
import numpy as np


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, cols_data: list[str], cols_gt: list[str], seq_len: int=128):
        assert isinstance(df, pd.DataFrame)
        assert isinstance(cols_data, list) and len(cols_data) > 0
        assert isinstance(cols_gt,   list) and len(cols_gt) > 0
        assert isinstance(seq_len,   int)  and seq_len > 0
        self.seq_len   = seq_len
        self.cols_data = cols_data
        self.cols_gt   = cols_gt
        self.data      = df[self.cols_data].to_numpy().astype(np.float32)
        self.gt        = df[self.cols_gt  ].to_numpy().astype(np.int64)
        self.indexes   = np.arange(seq_len, len(self.data), seq_len // 4)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx: int):
        _idx = self.indexes[idx]
        return torch.tensor(self.data[_idx - self.seq_len:_idx]), torch.tensor(self.gt[_idx])