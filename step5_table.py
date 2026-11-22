import csv
import json

S_F = 25
fail = {}
for r in csv.reader(open("results/loss_ft.csv")):
    fail[int(r[1])] = float(r[3])        # для повторених кроків лишається останній запис
base = {int(r["step"]): float(r["loss"]) for r in csv.DictReader(open("results/loss_ddp.csv"))}
res = json.load(open("results/step5_result.json", encoding="utf-8"))
acc_base = json.load(open("results/step2_train.json", encoding="utf-8"))["ACC"]

print("Таблиця 6. Відмова і відновлення")
print("K_c | s_f | крок останньої контрольної точки | кроків повторно | t_lost, с")
print(f"{res['K_c']} | {res['s_f']} | {res['last_ckpt_step']} | {res['steps_redone']} | {res['t_lost_s']}")
print(f"(t_f − t_c = {res['t_f_minus_t_c_s']} с, t_r = {res['t_r_s']} с)\n")

ks = range(S_F - 2, S_F + 3)
print("Крок | " + " | ".join(str(k) for k in ks) + " | ACC після n_ep епох")
print("Втрата, з відмовою | " + " | ".join(f"{fail[k]:.6f}" for k in ks) + f" | {res['ACC']:.4f}")
print("Втрата, без відмови | " + " | ".join(f"{base[k]:.6f}" for k in ks) + f" | {acc_base:.4f}")
print(f"\nНайбільша різниця втрати за всі кроки: {max(abs(fail[k] - base[k]) for k in base):.2e}")