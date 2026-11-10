"""1D-CNN over a tabular feature vector (re-implementation of the V1/V2 model).

The convolution slides over the *column order* of the preprocessed feature
vector. Because tabular columns have no natural order, we test whether the
learned performance depends on an arbitrary permutation of that order.
PyTorch is optional: the rest of the pipeline and the Streamlit app run without it.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn
    TORCH_OK = True
except ImportError:  # pragma: no cover
    TORCH_OK = False

from sklearn.metrics import average_precision_score


if TORCH_OK:
    class _Net(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=2), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=2), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Flatten(), nn.Dropout(0.25),
                nn.Linear(64 * (n_features - 2), 32), nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.body(x.unsqueeze(1)).squeeze(1)


class TabularCNN:
    """Sklearn-style wrapper. Input must already be numeric (one-hot + scaled)."""

    def __init__(self, epochs=12, batch_size=1024, lr=1e-3, patience=3, seed=42, order=None):
        self.epochs, self.batch_size, self.lr, self.patience = epochs, batch_size, lr, patience
        self.seed, self.order = seed, order

    def _x(self, X):
        X = np.asarray(X, dtype=np.float32)
        return X[:, self.order] if self.order is not None else X

    def fit(self, X, y, X_val, y_val):
        if not TORCH_OK:
            raise RuntimeError("PyTorch is not installed; pip install torch")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(4)
        X, Xv = self._x(X), self._x(X_val)
        y = np.asarray(y, dtype=np.float32)
        self.net_ = _Net(X.shape[1])
        pos = max(y.sum(), 1.0)
        pos_weight = torch.tensor(np.sqrt((len(y) - pos) / pos), dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, bad = -1.0, None, 0
        self.history_ = []
        for ep in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), self.batch_size):
                b = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net_(Xt[b]), yt[b])
                loss.backward()
                opt.step()
            ap = average_precision_score(y_val, self._predict(Xv))
            self.history_.append(ap)
            if ap > best:
                best, bad = ap, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        return self

    def _predict(self, X):
        self.net_.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(X), 8192):
                out.append(torch.sigmoid(self.net_(torch.from_numpy(X[i:i + 8192]))).numpy())
        return np.concatenate(out)

    def predict_proba(self, X):
        p = self._predict(self._x(X))
        return np.column_stack([1 - p, p])
