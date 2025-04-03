import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=20, forecast_horizon=12):
        
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.data = data
        self.X, self.Y = self.create_sequences(data, seq_len, forecast_horizon)
    
    def create_sequences(self, data, seq_len, forecast_horizon):
        X, Y = [], []
        for i in range(len(data) - seq_len - forecast_horizon + 1):
            X.append(data[i:i+seq_len])
            Y.append(data[i+seq_len:i+seq_len+forecast_horizon])
        return np.array(X), np.array(Y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)