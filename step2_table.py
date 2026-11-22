import json


def rd(n):
    with open(f"results/step2_{n}.json", encoding="utf-8") as f:
        return json.load(f)


s1, man, ddp, ns, eq, tr = (json.load(open("results/step1.json", encoding="utf-8")),
                            rd("manual"), rd("ddp"), rd("ddp_ns"), rd("equiv"), rd("train"))
t_comm = ddp["t_step_ms"] - ns["t_step_ms"]
c = t_comm / ddp["t_step_ms"]
print(f"1 процес:        t_step = {s1['t_step_ms']} мс, X = {s1['X1']}")
print(f"ручна синхр.:    t_step = {man['t_step_ms']} мс, t_sync = {man['t_sync_ms']} мс, "
      f"частка = {man['t_sync_ms'] / man['t_step_ms']:.3f}, X = {man['X']}")
print(f"DDP:             t_step = {ddp['t_step_ms']} мс, t_comm,exp = {t_comm:.2f} мс, "
      f"c = {c:.3f}, X = {ddp['X']}")
print(f"DDP no_sync:     t_step,ns = {ns['t_step_ms']} мс, X = {ns['X']}")
print(f"max|g-ḡ|/max|g| = {eq['max_grad_ratio']:.3e}, різниця сум ваг = {eq['sums_diff']:.3e}")
print(f"ACC: 1 процес {s1['ACC']:.4f}, DDP {tr['ACC']:.4f}; AUC: 1 процес {s1['AUC']:.4f}, DDP {tr['AUC']:.4f}")