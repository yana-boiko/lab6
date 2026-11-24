# Лабораторна робота 6. Розподілене навчання нейромережі за даними на CPU

Варіант 4: OrganCMNIST (Д4), ширина мережі Ш1 (16 / 32 / 128), P_max = 3, B = 192, K_c = 10, s_f = 25.
PyTorch (CPU), бекенд gloo, запуск через torchrun, DistributedDataParallel.

**Звіт:** [report/report.md](report/report.md)


## Підготовка

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -c "import os; from medmnist import OrganCMNIST; os.makedirs('data', exist_ok=True); OrganCMNIST(split='train', download=True, root='data')"
```

## Запуск

На Windows torchrun запускається через обгортку `run_torchrun.py` (збірка PyTorch без libuv).

```
python step1_single.py                                                           # крок 1: один процес
python run_torchrun.py --standalone --nproc_per_node=3 step2_dist.py --mode equiv   # крок 2: еквівалентність
python run_torchrun.py --standalone --nproc_per_node=3 step2_dist.py --mode manual  # ручна синхронізація
python run_torchrun.py --standalone --nproc_per_node=3 step2_dist.py --mode ddp     # DDP
python run_torchrun.py --standalone --nproc_per_node=3 step2_dist.py --mode ddp_ns  # DDP, no_sync
python run_torchrun.py --standalone --nproc_per_node=3 step2_dist.py --mode train   # навчання n_ep епох
python run_torchrun.py --standalone --nproc_per_node=3 --max-restarts=1 step5_ft.py # крок 5: відмова і відновлення
python step2_table.py && python step3_table.py && python step4_table.py && python step5_table.py
```

Для кроків 3–4 режими `ddp` і `ddp_ns` запускаються з іншою кількістю процесів, параметрами `--b`, `--lr` і `--tag` (див. звіт).

## Структура

- `common.py` — параметри варіанта, завантаження даних, мережа, оцінка ACC і AUC
- `step1_single.py` — навчання в одному процесі, n_params, M, X(1)
- `step2_dist.py` — ручна синхронізація, DDP, перевірка еквівалентності, вимір кроку, навчання
- `step5_ft.py` — контрольні точки, імітація відмови, відновлення
- `step2_table.py`, `step3_table.py`, `step4_table.py`, `step5_table.py` — таблиці і графіки
- `run_torchrun.py` — обгортка для запуску torchrun на Windows
- `results/` — сирі результати (json, csv, графік)
- `ЛР6_звіт_Бойко.docx` — звіт