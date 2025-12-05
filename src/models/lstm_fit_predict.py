# src/models/lstm_fit_predict.py
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def create_sequences(values, lookback=14):
    X, y = [], []
    for i in range(len(values) - lookback):
        X.append(values[i:i+lookback])
        y.append(values[i+lookback])
    return np.array(X), np.array(y)


def fit_predict_lstm_fixed(train, valid, date_col, target_col, lookback=14,
                           hidden_size=64, epochs=50, lr=0.001):

    # ----- 1) Preparação -----
    train_vals = train[target_col].values.reshape(-1, 1)
    valid_vals = valid[target_col].values.reshape(-1, 1)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_vals)
    valid_scaled = scaler.transform(valid_vals)

    X_train, y_train = create_sequences(train_scaled, lookback)
    X_valid, _ = create_sequences(np.concatenate([train_scaled[-lookback:],
                                                 valid_scaled]), lookback)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)

    model = LSTMModel(input_size=1, hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

    # ----- 2) Treino simples -----
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    # ----- 3) Previsão -----
    model.eval()
    pred_scaled = model(X_valid).detach().numpy()
    preds = scaler.inverse_transform(pred_scaled)

    # Os últimos N preds correspondem exatamente ao período de validação
    return preds[-len(valid):].flatten(), model


