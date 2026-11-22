import json

import matplotlib.pyplot as plt

from common import B, P_MAX

rows = []
for P in range(1, P_MAX + 1):
    d = json.load(open(f"results/step2_ddp_P{P}.json", encoding="utf-8"))
    n = json.load(open(f"results/step2_ddp_ns_P{P}.json", encoding="utf-8"))
    rows.append((P, B // P, d["t_step_ms"], n["t_step_ms"], d["X"]))

X1 = rows[0][4]
out = []
print("P | b | t_step, мс | t_step,ns, мс | X, прикл./с | S | E | e | c")
for P, b, ts, tns, X in rows:
    S = X / X1
    E = S / P
    e = (1 / S - 1 / P) / (1 - 1 / P) if P > 1 else None
    c = (ts - tns) / ts
    out.append({"P": P, "b": b, "t_step_ms": ts, "t_step_ns_ms": tns, "X": X,
                "S": round(S, 3), "E": round(E, 3),
                "e": None if e is None else round(e, 3), "c": round(c, 3)})
    print(f"{P} | {b} | {ts} | {tns} | {X} | {S:.3f} | {E:.3f} | "
          f"{'−' if e is None else f'{e:.3f}'} | {c:.3f}")

json.dump(out, open("results/step3_table.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

Ps = [r[0] for r in rows]
plt.figure(figsize=(7, 4.5))
plt.plot(Ps, [r[4] for r in rows], "o-", label="виміряне X(P)")
plt.plot(Ps, [P * X1 for P in Ps], "--", color="gray", label="ідеал P · X(1)")
plt.xticks(Ps)
plt.xlabel("Кількість процесів P")
plt.ylabel("X, прикл./с")
plt.title("Strong scaling, B = 192")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig("results/strong_scaling.png", dpi=150, bbox_inches="tight")
print("Збережено: results/strong_scaling.png")