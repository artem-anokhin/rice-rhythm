#!/usr/bin/env python3
"""Распределения для подбора порогов: падения HH между соседними датами, максимум и размах HV.

  python thresholds.py out          → out/thresholds.png + цифры в консоль
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CREDIT = "NISAR data courtesy of NASA/JPL, ASF DAAC"
out = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
df = pd.read_csv(out / "timeseries.csv", parse_dates=["date"])
hh = df.pivot(index="field", columns="date", values="HH_db")
hv = df.pivot(index="field", columns="date", values="HV_db")

hv_max = hv.max(axis=1)
hv_rng = hv.max(axis=1) - hv.min(axis=1)
drops = -(hh.diff(axis=1).iloc[:, 1:])          # >0 — HH упал
cand = (hv_max >= float(sys.argv[2]) if len(sys.argv) > 2 else hv_max > -99)
d_all = drops.values[np.isfinite(drops.values)]
d_rice = drops[cand].values
d_rice = d_rice[np.isfinite(d_rice)]

def q(a, ps=(50, 75, 90, 95, 99)):
    return ", ".join(f"p{p}={np.percentile(a, p):.2f}" for p in ps)

print(f"полей: {len(hh)}, дат: {hh.shape[1]}")
print(f"HV max по полям: {q(hv_max.dropna(), (5, 25, 50, 75, 95))}")
print(f"HV размах по полям: {q(hv_rng.dropna(), (5, 25, 50, 75, 95))}")
print(f"падения HH (все пары дат, только >0): {q(d_all[d_all > 0])}")
# Шумовой уровень: разность соседних дат у «стабильных» полей (малый размах HV)
stable = hv_rng < hv_rng.quantile(0.2)
dn = hh.diff(axis=1).iloc[:, 1:][stable].values
dn = np.abs(dn[np.isfinite(dn)])
print(f"шум |ΔHH| у стабильных полей (нижние 20% размаха HV): {q(dn)}")
# по парам дат: медиана изменения — видно общесезонные скачки (влажность, ветер)
med = hh.diff(axis=1).iloc[:, 1:].median()
print("медианный ΔHH по парам дат: " + ", ".join(f"{d:%d.%m}:{v:+.1f}" for d, v in med.items()))

fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), dpi=110)
axs[0].hist(d_all[d_all > 0], bins=80, color="#c2185b", alpha=0.8, label="все поля")
axs[0].hist(dn, bins=80, color="#555", alpha=0.5, label="шум (стабильные поля)")
axs[0].set_yscale("log"); axs[0].set_title("Падение HH между соседними датами, дБ"); axs[0].legend()
axs[1].hist(hv_max.dropna(), bins=60, color="#2e7d32"); axs[1].set_title("Максимум HV за сезон, дБ")
axs[2].hist(hv_rng.dropna(), bins=60, color="#1f6fd1"); axs[2].set_title("Размах HV за сезон, дБ")
fig.text(0.995, 0.005, CREDIT, ha="right", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(out / "thresholds.png")
print(f"→ {out / 'thresholds.png'}")
