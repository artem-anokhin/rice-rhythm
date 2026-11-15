#!/usr/bin/env python3
"""Проверка «сброса воды» другими треками NISAR.

Массив из 8 соседних квадратов (pulse) сравниваем с остальным рисом (flat) по каждому треку.
Сравнение «массив минус остальной рис» в одной и той же съёмке убирает разницу калибровки и угла между треками.
  python track_check.py out data/stack.nc:078D data_kzyl_t150D/stack.nc:150D ...
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as stransform

warnings.filterwarnings("ignore")
CLUSTER = ["C054_046", "C055_046", "C055_047", "C056_045", "C056_046", "C057_044", "C057_045", "C057_046"]
CREDIT = "NISAR data courtesy of NASA/JPL, ASF DAAC"


def cell_values(stack_path, feats):
    st = xr.open_dataset(stack_path, engine="h5netcdf").load()
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{int(st.attrs['epsg'])}", always_xy=True).transform
    x, y = st.x.values, st.y.values
    hh = 10 * np.log10(np.where(st.HH.values > 0, st.HH.values, np.nan))
    rows = []
    for fid, lab, g in feats:
        x0, y0, x1, y1 = stransform(tr, g).bounds
        ix = (x >= x0) & (x <= x1); iy = (y >= y0) & (y <= y1)
        if ix.sum() < 3 or iy.sum() < 3:
            continue
        sub = hh[:, iy][:, :, ix]
        for t in range(len(st.time)):
            v = sub[t]
            if np.isfinite(v).mean() < 0.5:
                continue
            rows.append({"field": fid, "label": lab, "date": pd.Timestamp(st.time.values[t]).normalize(),
                         "HH_db": float(np.nanmedian(v))})
    return pd.DataFrame(rows)


def main():
    out = Path(sys.argv[1])
    gj = json.loads((out / "fields_result.geojson").read_text())
    feats = []
    for f in gj["features"]:
        fid, lab = f["properties"]["id"], f["properties"]["label"]
        if fid in CLUSTER:
            feats.append((fid, "cluster", shape(f["geometry"])))
        elif lab == "flat":
            feats.append((fid, "flat", shape(f["geometry"])))
    res = []
    for arg in sys.argv[2:]:
        p, name = arg.split(":")
        if not Path(p).exists():
            print(f"{name}: нет {p}"); continue
        df = cell_values(p, feats)
        if df.empty:
            print(f"{name}: участок не покрыт"); continue
        g = df.groupby(["date", "label"]).HH_db.median().unstack()
        n = df.groupby(["date", "label"]).field.nunique().unstack()
        g = g[(n.get("cluster", 0) >= 6) & (n.get("flat", 0) >= 50)]
        g["diff"] = g["cluster"] - g["flat"]
        g["track"] = name
        res.append(g.reset_index())
        print(f"\nтрек {name}: (массив − остальной рис), дБ")
        for _, r in g.reset_index().iterrows():
            print(f"  {r.date:%d.%m}  массив {r.cluster:6.1f}  рис {r.flat:6.1f}  разница {r['diff']:+5.1f}")
    allr = pd.concat(res)
    allr.to_csv(out / "track_check.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_axes([0.11, 0.13, 0.86, 0.66])
    style = {"078D": ("#c2185b", "o", 5), }
    cols = ["#1565c0", "#2e7d32", "#6a1b9a"]
    k = 0
    for name, g in allr.groupby("track", sort=False):
        g = g[(g.date >= "2026-08-25") & (g.date <= "2026-10-15")]
        if name in style:
            c, m, lw = style[name]; lab = f"{name} — основной трек"
        else:
            c, m, lw = cols[k % 3], "s", 3.5; k += 1; lab = f"{name} — " + ("другой нисходящий трек" if name.endswith("D") else "восходящий трек")
        ax.plot(g.date, g["diff"], "-" + m, color=c, lw=lw, ms=14, label=lab)
    ax.axhline(0, color="#555", lw=1.5)
    ax.axvspan(pd.Timestamp("2026-09-12"), pd.Timestamp("2026-10-04"), color="#f2a900", alpha=0.12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.tick_params(labelsize=25)
    ax.set_ylabel("массив минус остальной рис, дБ\n(ниже нуля = темнее соседей)", fontsize=23)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=22, loc="upper left", framealpha=0.95)
    fig.text(0.11, 0.9, "Сброс воды: что видят другие треки NISAR", fontsize=36, weight="bold")
    fig.text(0.11, 0.84, "8 соседних квадратов в сравнении с остальным рисом · жёлтое — окно провала 11.09–05.10", fontsize=23)
    fig.text(0.99, 0.015, CREDIT, ha="right", va="bottom", fontsize=21, color="#444")
    fig.savefig(out / "track_check.png", dpi=100)
    print("→", out / "track_check.png")


if __name__ == "__main__":
    main()
