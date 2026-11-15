#!/usr/bin/env python3
"""Слепой тест на Арканзасе (Humnoke, AmeriFlux US-HRA и US-HRC).

Тот же classify() из analyze_fields.py и те же пороги, что в Кызылорде (DROP 2.4, RISE 1.3, рост HV 3.0).
Ряды полей — окно 3×3 пикселя в координатах станции; общий сдвиг по датам — медиана квадратов 300 м по AOI,
как в основном алгоритме.
  python ark_test.py data_arkansas/stack.nc out_arkansas 084D
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

import analyze_fields as af

warnings.filterwarnings("ignore")
af.DROP_DB, af.RISE_DB = 2.4, 1.3          # пороги Кызылорды, без подгонки
STATIONS = {"US-HRA (поле A, где применяли AWD)": (34.5852, -91.7517),
            "US-HRC (поле C, постоянное затопление)": (34.5888, -91.7517)}
COLORS = ["#f2a900", "#1f6fd1"]


def main():
    src, out, name = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    out.mkdir(exist_ok=True)
    st = xr.open_dataset(src, engine="h5netcdf").load()
    dates = [pd.Timestamp(t) for t in st.time.values]
    early = np.array([(d.month, d.day) <= af.EARLY_END for d in dates])
    cells = af.grid_series(st)
    HH = np.array([c[2] for c in cells]); HV = np.array([c[3] for c in cells])
    common = np.nanmedian(HH, axis=0); common = common - np.nanmean(common)
    lab = [af.classify(h - common, v, early)[0] for h, v in zip(HH, HV)]
    bg = pd.Series(lab).value_counts().to_dict()

    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{int(st.attrs['epsg'])}", always_xy=True)
    x, y = st.x.values, st.y.values
    lines = [f"Трек {name}: {len(dates)} дат {dates[0]:%d.%m}–{dates[-1]:%d.%m.%Y}; пороги DROP {af.DROP_DB}, RISE {af.RISE_DB}",
             f"Фон, квадраты 300 м по AOI: {bg}",
             "Общий сдвиг по датам (медиана AOI), дБ: " + " ".join(f"{d:%d.%m}:{c:+.1f}" for d, c in zip(dates, common))]
    res = {}
    for (sname, (lat, lon)) in STATIONS.items():
        px, py = tr.transform(lon, lat)
        i, j = int(np.argmin(abs(x - px))), int(np.argmin(abs(y - py)))
        win = lambda v: st[v].values[:, j - 1:j + 2, i - 1:i + 2].reshape(len(dates), -1)
        hh = af.to_db(np.nanmedian(np.where(win("HH") > 0, win("HH"), np.nan), axis=1))
        hv = af.to_db(np.nanmedian(np.where(win("HV") > 0, win("HV"), np.nan), axis=1))
        label, info = af.classify(hh - common, hv, early)
        drops = [k for k in range(1, len(hh)) if np.isfinite(hh[k - 1] - hh[k]) and (hh[k - 1] - common[k - 1]) - (hh[k] - common[k]) >= af.DROP_DB and not early[k - 1]]
        res[sname] = (hh, hv, label, info, drops)
        lines.append(f"\n{sname}: метка «{af.LABELS[label][0] if label in af.LABELS else label}»; "
                     f"сбросы (падение+возврат): {[f'{dates[k]:%d.%m}' for k in info['pulses']]}; "
                     f"уборка по HV: {dates[info['harvest_idx']]:%d.%m}" if info.get('harvest_idx') is not None else
                     f"\n{sname}: метка «{af.LABELS[label][0] if label in af.LABELS else label}»; "
                     f"сбросы: {[f'{dates[k]:%d.%m}' for k in info['pulses']]}; уборка по HV не найдена")
        lines.append(f"  все падения HH ≥ {af.DROP_DB} дБ после 15.07 (с вычетом общего сдвига): {[f'{dates[k]:%d.%m}' for k in drops]}")
        lines.append("  HH: " + " ".join(f"{d:%d.%m}:{v:.1f}" for d, v in zip(dates, hh)))
        lines.append("  HV: " + " ".join(f"{d:%d.%m}:{v:.1f}" for d, v in zip(dates, hv)))
    txt = "\n".join(lines)
    print(txt)
    (out / f"positive_control_{name}.txt").write_text(txt + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_axes([0.11, 0.13, 0.86, 0.66])
    for c, (sname, (hh, hv, label, info, drops)) in zip(COLORS, res.items()):
        lbl = f"{sname}: алгоритм — «{'сброс воды' if label == 'pulse' else 'вода весь сезон' if label == 'flat' else 'не рис'}»"
        ax.plot(dates, hh, "-o", color=c, lw=4, ms=13, label=lbl)
        for k in info["pulses"]:
            ax.annotate("сброс", (dates[k], hh[k]), xytext=(0, -55), textcoords="offset points", ha="center",
                        fontsize=22, color=c, weight="bold", arrowprops=dict(arrowstyle="->", color=c, lw=3))
        if info.get("harvest_idx") is not None:
            ax.axvline(dates[info["harvest_idx"]], color=c, ls=":", lw=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.tick_params(labelsize=24)
    ax.set_ylabel("сигнал радара HH, дБ\n(выше = ярче отражение)", fontsize=23)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=20, loc="upper right", framealpha=0.95)
    fig.text(0.11, 0.9, "Слепая проверка: две научные станции в Арканзасе", fontsize=36, weight="bold")
    fig.text(0.11, 0.84, f"Те же правила и пороги, что в Кызылорде · трек NISAR {name} · пунктир — уборка по HV",
             fontsize=22)
    fig.text(0.99, 0.015, af.CREDIT, ha="right", va="bottom", fontsize=21, color="#444")
    fig.savefig(out / f"positive_control_{name}.png", dpi=100)
    print("→", out / f"positive_control_{name}.png")


if __name__ == "__main__":
    main()
