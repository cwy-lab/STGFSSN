import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader.dataset import TimeSeriesDataset
from model.model import GraphMambaImproved

def main():
    data_df = pd.read_csv('data.csv')
    data = data_df.values.astype(np.float32)  # (total_steps, num_features)

    
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)

    
    data_norm = (data - data_min) / (data_max - data_min)

    
    seq_len = 80
    forecast_horizon = 20  
    dataset = TimeSeriesDataset(data_norm, seq_len, forecast_horizon)

    
    total_samples = len(dataset)
    test_ratio = 0.2
    train_samples = int(total_samples * (1 - test_ratio))
    train_data = torch.utils.data.Subset(dataset, list(range(train_samples)))
    test_data = torch.utils.data.Subset(dataset, list(range(train_samples, total_samples)))

    # DataLoader
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    
    num_nodes = data.shape[1]  
    hidden_dim = 64
    dropout_rate = 0.2
    num_heads = 4
    learning_rate = 1e-3
    num_epochs = 100

    model = GraphMambaImproved(num_nodes=num_nodes, seq_len=seq_len, hidden_dim=hidden_dim, 
                                forecast_horizon=forecast_horizon, dropout_rate=dropout_rate, num_heads=num_heads)

    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in train_loader:
            # inputs: (B, seq_len, num_nodes)
            # targets: (B, forecast_horizon, num_nodes) -> (B, num_nodes, forecast_horizon)
            inputs = inputs.to(device)
            targets = targets.permute(0, 2, 1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # (B, num_nodes, forecast_horizon)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    
    model.eval()
    def predict(loader):
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)  # (B, num_nodes, forecast_horizon)
                targets = targets.permute(0, 2, 1)
                preds_last = outputs[:, :, -1]   # (B, num_nodes)
                targets_last = targets[:, :, -1]   # (B, num_nodes)
                all_preds.append(preds_last.cpu().numpy())
                all_targets.append(targets_last.cpu().numpy())
        return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)

    train_preds, train_targets = predict(train_loader)
    test_preds, test_targets = predict(test_loader)

    
    train_preds = train_preds * (data_max - data_min) + data_min
    train_targets = train_targets * (data_max - data_min) + data_min
    test_preds = test_preds * (data_max - data_min) + data_min
    test_targets = test_targets * (data_max - data_min) + data_min

    
    rows = []
    for i in range(train_preds.shape[0]):
        row = {'set': 'train', 'index': i}
        for j in range(num_nodes):
            row[f'sensor{j}_true'] = train_targets[i, j]
            row[f'sensor{j}_pred'] = train_preds[i, j]
        rows.append(row)
    for i in range(test_preds.shape[0]):
        row = {'set': 'test', 'index': i}
        for j in range(num_nodes):
            row[f'sensor{j}_true'] = test_targets[i, j]
            row[f'sensor{j}_pred'] = test_preds[i, j]
        rows.append(row)
    result_df = pd.DataFrame(rows)
    result_df.to_csv("prediction_results.csv", index=False)
    

if __name__ == '__main__':
    main()
