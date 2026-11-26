"""1D-CNN and training loop, copied from notebooks/03_deep_models.ipynb, parameterised by dataset.

Same architecture, optimiser (Adam lr 1e-3), batch size 256, gradient clipping, early stopping on validation RMSE,
and target scaling by the RUL cap (removing the scaling makes the deep models collapse to the mean, see README).
"""
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CAP = 125.0


class CNN1D(nn.Module):
    def __init__(self, n_feat, W, drop=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(),
            nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(64 * W, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.conv(x.transpose(1, 2))).squeeze(-1)   # conv wants (batch, features, time)


def predict(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X).to(device)).cpu().numpy() * CAP


def train_cnn(d, seed, device="cpu", epochs=80, lr=1e-3, patience=10, min_epochs=15):
    """Train a 1D-CNN on windows dict d. Returns (best val RMSE, test predictions, seconds)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_tr, y_tr, X_val, y_val = d["X_tr"], d["y_tr"], d["X_val"], d["y_val"]
    model = CNN1D(X_tr.shape[2], X_tr.shape[1]).to(device)
    dl = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr / CAP)), batch_size=256, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best, best_state, wait, t0 = 1e9, None, 0, time.time()
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = float(np.sqrt(np.mean((y_val - predict(model, X_val, device)) ** 2)))
        if v < best:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience and ep >= min_epochs:
                break
    model.load_state_dict(best_state)
    return best, predict(model, d["X_test"], device), time.time() - t0
