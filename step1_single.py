import json
import os
import statistics
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from common import *

torch.set_num_threads(N_TH)

train = load_split(DATA, "train", n=N_TRAIN)
test = load_split(DATA, "test")
print(f"train = {len(train)}, test = {len(test)}, n_th = {N_TH}")

model = build_model()
n_params = sum(p.numel() for p in model.parameters())
assert n_params == n_params_formula(C, C1, C2, H, K)
M = 4 * n_params
print(f"n_params = {n_params}, M = {M} байт = {M / 2**20:.2f} МіБ")

loss_fn = nn.CrossEntropyLoss()


def make_loader():
    g = torch.Generator().manual_seed(0)
    return DataLoader(train, batch_size=B, shuffle=True, drop_last=True, generator=g)


opt = torch.optim.SGD(model.parameters(), lr=LR0, momentum=MOMENTUM)
loader = make_loader()
for epoch in range(N_EP):
    model.train()
    losses = []
    for x, y in loader:
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"епоха {epoch + 1}: середня втрата {sum(losses) / len(losses):.4f}, кроків {len(losses)}")
acc, auc = evaluate(model, test)
print(f"ACC = {acc:.4f}, AUC = {auc:.4f}")

model = build_model()
opt = torch.optim.SGD(model.parameters(), lr=LR0, momentum=MOMENTUM)
loader = make_loader()


def batches():
    while True:
        for b in loader:
            yield b


gen = batches()
model.train()


def step(x, y):
    opt.zero_grad()
    loss_fn(model(x), y).backward()
    opt.step()


for _ in range(10):
    step(*next(gen))
times = []
for _ in range(50):
    x, y = next(gen)
    t0 = time.perf_counter()
    step(x, y)
    times.append(time.perf_counter() - t0)
t_step = statistics.median(times)
t = sum(times)
X1 = 50 * B / t
print(f"t_step (медіана) = {t_step * 1000:.2f} мс, t(50 кроків) = {t:.3f} с, X(1) = {X1:.1f} прикл./с")

os.makedirs("results", exist_ok=True)
with open("results/step1.json", "w", encoding="utf-8") as f:
    json.dump({"n_th": N_TH, "n_params": n_params, "M_bytes": M, "M_MiB": round(M / 2**20, 3),
               "ACC": acc, "AUC": auc, "t_step_ms": round(t_step * 1000, 2),
               "X1": round(X1, 1)}, f, ensure_ascii=False, indent=2)