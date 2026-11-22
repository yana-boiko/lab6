import argparse
import csv
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["equiv", "manual", "ddp", "ddp_ns", "train"])
ap.add_argument("--tag", default="")
a = ap.parse_args()

torch.set_num_threads(N_TH)
_rank = int(os.environ["RANK"])
_world = int(os.environ["WORLD_SIZE"])
_store = dist.TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]),
                       _world, is_master=(_rank == 0), use_libuv=False)
dist.init_process_group(backend="gloo", store=_store, rank=_rank, world_size=_world)
rank, P = dist.get_rank(), dist.get_world_size()
b = B // P

train = load_split(DATA, "train", n=N_TRAIN)
loss_fn = nn.CrossEntropyLoss()
os.makedirs("results", exist_ok=True)


def new_model():
    m = build_model()
    for p in m.parameters():
        dist.broadcast(p.data, src=0)
    return m


def sync_grads(m):
    for p in m.parameters():
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= P


sampler = DistributedSampler(train, num_replicas=P, rank=rank, shuffle=True, seed=0)
loader = DataLoader(train, batch_size=b, sampler=sampler, drop_last=True)


def batches():
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def save(name, data):
    if rank == 0:
        with open(f"results/step2_{name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(json.dumps(data, ensure_ascii=False, indent=2))

if a.mode == "equiv":
    model = new_model()
    x_all, y_all = train.tensors[0][:B], train.tensors[1][:B]

    model.zero_grad()
    loss_fn(model(x_all), y_all).backward()
    g = [p.grad.clone() for p in model.parameters()]

    model.zero_grad()
    loss_fn(model(x_all[rank::P]), y_all[rank::P]).backward()
    sync_grads(model)
    ratio = max((gi - p.grad).abs().max().item() / gi.abs().max().item()
                for gi, p in zip(g, model.parameters()))

    model = new_model()
    opt = torch.optim.SGD(model.parameters(), lr=LR0, momentum=MOMENTUM)
    gen = batches()
    for _ in range(50):
        x, y = next(gen)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        sync_grads(model)
        opt.step()
    with torch.no_grad():
        s = torch.tensor([sum(p.double().sum().item() for p in model.parameters())],
                         dtype=torch.float64)
    out = [torch.zeros(1, dtype=torch.float64) for _ in range(P)]
    dist.all_gather(out, s)
    sums = [o.item() for o in out]
    save("equiv", {"max_grad_ratio": ratio, "weight_sums": sums,
                   "sums_diff": max(sums) - min(sums)})

elif a.mode in ("manual", "ddp", "ddp_ns"):
    model = new_model()
    if a.mode != "manual":
        model = DDP(model)
    opt = torch.optim.SGD(model.parameters(), lr=LR0, momentum=MOMENTUM)
    gen = batches()
    steps, syncs = [], []
    for i in range(60):
        x, y = next(gen)
        t0 = time.perf_counter()
        opt.zero_grad()
        if a.mode == "ddp_ns":
            with model.no_sync():
                loss_fn(model(x), y).backward()
        else:
            loss_fn(model(x), y).backward()
        if a.mode == "manual":
            ts = time.perf_counter()
            sync_grads(model)
            t_sync = time.perf_counter() - ts
        opt.step()
        t_step = time.perf_counter() - t0
        if i >= 10:
            steps.append(t_step)
            if a.mode == "manual":
                syncs.append(t_sync)
    res = {"P": P, "b": b, "t_step_ms": round(statistics.median(steps) * 1000, 2),
           "X": round(50 * B / sum(steps), 1)}
    if a.mode == "manual":
        res["t_sync_ms"] = round(statistics.median(syncs) * 1000, 2)
    save(a.mode + a.tag, res)

else:
    model = DDP(new_model())
    opt = torch.optim.SGD(model.parameters(), lr=LR0, momentum=MOMENTUM)
    rows, step = [], 0
    for epoch in range(N_EP):
        sampler.set_epoch(epoch)
        model.train()
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            l = loss.detach().clone()
            dist.all_reduce(l)
            l /= P
            step += 1
            rows.append((step, epoch + 1, l.item()))
    if rank == 0:
        with open("results/loss_ddp.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "epoch", "loss"])
            w.writerows(rows)
        test = load_split(DATA, "test")
        acc, auc = evaluate(model.module, test)
        save("train", {"steps": step, "ACC": acc, "AUC": auc})
    dist.barrier()

dist.destroy_process_group()