import json
import os
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from common import *

K_C, S_F = 10, 25
CKPT, CKPT_TMP, MARKER = "ckpt.pt", "ckpt.tmp", "fail.marker"
LOG = "results/loss_ft.csv"

torch.set_num_threads(N_TH)
rank = int(os.environ["RANK"])
P = int(os.environ["WORLD_SIZE"])
attempt = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))
store = dist.TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]),
                      P, is_master=(rank == 0), use_libuv=False)
store = dist.PrefixStore(f"attempt_{attempt}", store)  
dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=P)
b = B // P
os.makedirs("results", exist_ok=True)

train = load_split(DATA, "train", n=N_TRAIN)
loss_fn = nn.CrossEntropyLoss()

model = build_model()
opt = torch.optim.SGD(model.parameters(), lr=LR0, momentum=MOMENTUM)
epoch0, skip, step, t_c = 0, 0, 0, None
resumed = os.path.exists(CKPT)
if resumed:
    st = torch.load(CKPT)
    model.load_state_dict(st["model"])
    opt.load_state_dict(st["opt"])
    epoch0, skip, step, t_c = st["epoch"], st["step_in_epoch"], st["step"], st["t"]
    if rank == 0:
        print(f"[спроба {attempt}] відновлено з контрольної точки: крок {step}", flush=True)
ck_step = step
model = DDP(model)

sampler = DistributedSampler(train, num_replicas=P, rank=rank, shuffle=True, seed=0)
loader = DataLoader(train, batch_size=b, sampler=sampler, drop_last=True)
log = open(LOG, "a") if rank == 0 else None
wait_first = resumed
t_first = None

for epoch in range(epoch0, N_EP):
    sampler.set_epoch(epoch)
    model.train()
    for i, (x, y) in enumerate(loader):
        if epoch == epoch0 and i < skip:
            continue
        if rank == P - 1 and step + 1 == S_F and not os.path.exists(MARKER):
            with open(MARKER, "w") as f:
                json.dump({"t_f": time.time(), "step": S_F}, f)
            os._exit(1)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        l = loss.detach().clone()
        dist.all_reduce(l)
        l /= P
        step += 1
        if rank == 0:
            log.write(f"{attempt},{step},{epoch + 1},{l.item()}\n")
            log.flush()
            if wait_first:
                t_first = time.time()
                wait_first = False
        if step % K_C == 0:
            if rank == 0:
                state = {"model": model.module.state_dict(), "opt": opt.state_dict(),
                         "epoch": epoch, "step_in_epoch": i + 1, "step": step,
                         "t": time.time()}
                torch.save(state, CKPT_TMP)
                os.replace(CKPT_TMP, CKPT)
            dist.barrier()

if rank == 0:
    acc, auc = evaluate(model.module, load_split(DATA, "test"))
    res = {"K_c": K_C, "s_f": S_F, "steps_total": step, "ACC": acc, "AUC": auc}
    if resumed and os.path.exists(MARKER):
        t_f = json.load(open(MARKER))["t_f"]
        res.update({"last_ckpt_step": ck_step,
                    "steps_redone": (S_F - 1) - ck_step,
                    "t_f_minus_t_c_s": round(t_f - t_c, 2),
                    "t_r_s": round(t_first - t_f, 2),
                    "t_lost_s": round((t_f - t_c) + (t_first - t_f), 2)})
    with open("results/step5_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
dist.barrier()
dist.destroy_process_group()