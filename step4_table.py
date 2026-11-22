import csv
import json

from common import B, P_MAX

b = B // P_MAX

print("Таблиця 4. Weak scaling, b =", b)
print("P | B_P | t_step, мс | X, прикл./с | S_w | E_w")
rows = []
for P in range(1, P_MAX + 1):
    name = f"results/step2_ddp_w{P}.json" if P < P_MAX else f"results/step2_ddp_P{P}.json"
    d = json.load(open(name, encoding="utf-8"))
    rows.append((P, d["t_step_ms"], d["X"]))
X1 = rows[0][2]
out = []
for P, ts, X in rows:
    Sw = X / X1
    out.append({"P": P, "B_P": P * b, "t_step_ms": ts, "X": X,
                "S_w": round(Sw, 3), "E_w": round(Sw / P, 3)})
    print(f"{P} | {P * b} | {ts} | {X} | {Sw:.3f} | {Sw / P:.3f}")
json.dump(out, open("results/step4_weak.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def last_epoch_max(path):
    r = list(csv.DictReader(open(path)))
    last = max(int(x["epoch"]) for x in r)
    return len(r), max(float(x["loss"]) for x in r if int(x["epoch"]) == last)


modes = [
    ("P = 1", b, 0.02, "results/step2_train_P1_eta0.json", "results/loss_ddp_P1_eta0.csv"),
    ("P = 3, з кроку 2", B, 0.02, "results/step2_train.json", "results/loss_ddp.csv"),
    ("P = 3", B, 0.06, "results/step2_train_P3_etaP.json", "results/loss_ddp_P3_etaP.csv"),
]
print("\nТаблиця 5. Якість за різних глобальних батчів")
print("Режим | B_P | η | кроків | ACC | AUC | найбільша втрата за останню епоху")
for name, bp, lr, jf, cf in modes:
    j = json.load(open(jf, encoding="utf-8"))
    steps, mx = last_epoch_max(cf)
    print(f"{name} | {bp} | {lr} | {steps} | {j['ACC']:.4f} | {j['AUC']:.4f} | {mx:.4f}")