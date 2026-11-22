import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

NAME, K, C = "organcmnist", 11, 1
C1, C2, H = 16, 32, 128
N_TRAIN = 7000
B = 192
P_MAX = 3
LR0 = 0.02
MOMENTUM = 0.9
N_EP = 3
C_PHYS = 4
N_TH = max(1, C_PHYS // P_MAX) 
DATA = f"data/{NAME}.npz"


def load_split(path, split, n=None):
    d = np.load(path)
    x = d[f"{split}_images"][:n]
    y = d[f"{split}_labels"][:n].reshape(-1)
    if x.ndim == 3:
        x = x[:, None, :, :]
    else:
        x = x.transpose(0, 3, 1, 2)
    x = torch.from_numpy(np.ascontiguousarray(x)).float()
    x = (x / 255 - 0.5) / 0.5
    return TensorDataset(x, torch.from_numpy(y.astype(np.int64)))


class Net(nn.Module):
    def __init__(self, C, c1, c2, h, K):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(C, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(7 * 7 * c2, h), nn.ReLU(),
            nn.Linear(h, K),
        )

    def forward(self, x):
        return self.head(self.features(x))


def n_params_formula(C, c1, c2, h, K):
    return 9 * C * c1 + c1 + 9 * c1 * c2 + c2 + 49 * c2 * h + h + h * K + K


def build_model():
    torch.manual_seed(0)
    return Net(C, C1, C2, H, K)


@torch.no_grad()
def evaluate(model, test):
    model.eval()
    probs, ys = [], []
    for x, y in DataLoader(test, batch_size=1024):
        probs.append(torch.softmax(model(x), dim=1))
        ys.append(y)
    prob = torch.cat(probs).numpy()
    y = torch.cat(ys).numpy()
    acc = float((prob.argmax(axis=1) == y).mean())
    auc = float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
    return acc, auc