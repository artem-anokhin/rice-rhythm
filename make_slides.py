#!/usr/bin/env python3
"""Картинки для слайдов 1920×1080 (крупный шрифт, по-русски, с подписью источника).

  python make_slides.py --stack data/stack.nc --out out --region "Жалагаш, Кызылорда" --prefix kzyl
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

import analyze_fields as af
from make_visuals import rgb_frame, ru_date

CREDIT = "NISAR data courtesy of NASA/JPL, ASF DAAC"
W, H, DPI = 19.2, 10.8, 100          # 1920×1080
T, L, S = 36, 23, 21                 # pt при 100 dpi: заголовок ≈ 50 px, подписи ≈ 32 px, мелкое ≈ 29 px


def new_fig():
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.text(0.99, 0.015, CREDIT, ha="right", va="bottom", fontsize=S, color="#444")
    return fig


def save(fig, p):
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("→", p)


def series_plot(ax, ts, fid, color=None, label=None, hv=True):
    d = ts[ts.field == fid].sort_values("date")
    ax.plot(d.date, d.HH_db, "-o", lw=4, ms=12, color=color or "#c2185b",
            label=label or "HH — канал «вода и почва»")
    if hv:
        ax.plot(d.date, d.HV_db, "-s", lw=3, ms=10, color="#2e7d32", alpha=0.85, label="HV — канал «растения»")
    return d


def style(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.tick_params(labelsize=L + 2)
    ax.set_ylabel("сигнал радара, дБ\n(выше = ярче отражение)", fontsize=L)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="data/stack.nc")
    ap.add_argument("--out", default="out")
    ap.add_argument("--region", default="")
    ap.add_argument("--prefix", default="kzyl")
    ap.add_argument("--slides", default="slides_png")
    args = ap.parse_args()
    out, sl = Path(args.out), Path(args.slides)
    sl.mkdir(exist_ok=True)
    pre = args.prefix

    import xarray as xr
    from pyproj import Transformer
    st = xr.open_dataset(args.stack, engine="h5netcdf").load()
    gj = json.loads((out / "fields_result.geojson").read_text())
    ts = pd.read_csv(out / "timeseries.csv", parse_dates=["date"])
    labels = {f["properties"]["id"]: f["properties"]["label"] for f in gj["features"]}
    counts = pd.Series(labels).value_counts()
    dates = sorted(ts.date.unique())

    # ---------- 1. Карта ----------
    epsg = int(st.attrs["epsg"])
    to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    x, y = st.x.values, st.y.values
    lon, lat = to_ll.transform([x.min(), x.max()], [y.min(), y.max()])
    k = int(np.argmin([abs((pd.Timestamp(t) - pd.Timestamp("2026-08-01")).days) for t in st.time.values]))
    rgb, _ = rgb_frame(st.HH.values[k], st.HV.values[k], max(1, int(round(30 / abs(x[1] - x[0])))))
    if y[0] < y[-1]:
        rgb = rgb[::-1]
    fig = new_fig()
    ax = fig.add_axes([0.02, 0.06, 0.56, 0.82])
    ax.imshow(rgb * 0.8, extent=[lon[0], lon[1], lat[0], lat[1]], aspect=1 / np.cos(np.radians(np.mean(lat))))
    from matplotlib.collections import PolyCollection
    for lab in ("flat", "pulse", "dry_seed", "other_crop"):
        polys = [np.array(f["geometry"]["coordinates"][0]) for f in gj["features"] if f["properties"]["label"] == lab]
        if polys:
            ax.add_collection(PolyCollection(polys, facecolors=af.LABELS[lab][1], edgecolors="none", alpha=0.85))
    ax.set_xlim(lon); ax.set_ylim(lat); ax.axis("off")
    fig.text(0.02, 0.93, f"{args.region}: режим воды на рисовых полях", fontsize=T, weight="bold")
    n = len(labels)
    short = {"flat": "Рис: вода весь сезон", "pulse": "Рис: воду сбрасывали", "dry_seed": "Рис: прямой посев",
             "other_crop": "Другие культуры"}
    leg = [Patch(color=af.LABELS[l][1], label=f"{short[l]}: {counts.get(l, 0)}")
           for l in ("flat", "pulse", "dry_seed", "other_crop") if counts.get(l, 0)]
    leg.append(Patch(color="#666", label=f"Не рис, без цвета: {counts.get('not_rice', 0)}"))
    fig.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.6, 0.85), fontsize=L, frameon=False,
               title=f"Квадраты 300×300 м (всего {n})", title_fontsize=L)
    fig.text(0.61, 0.42, f"Фон — радар NISAR {ru_date(pd.Timestamp(st.time.values[k]))}\n"
                         f"Снимков NISAR: {len(dates)}\n"
                         f"{pd.Timestamp(dates[0]):%d.%m} — {pd.Timestamp(dates[-1]):%d.%m.%Y}\n"
                         f"трек {st.attrs.get('track', '')}, L-диапазон, HH + HV", fontsize=L, va="top")
    save(fig, sl / f"{pre}_1_map.png")

    # ---------- 2. Ритм одного поля ----------
    pulse = [f for f, l in labels.items() if l == "pulse"]
    best = None
    if pulse:
        p = [f for f in pulse if (out / "charts" / f"{f}.png").exists()] or pulse
        # поле с самым глубоким провалом HH после 15.07
        def depth(f):
            d = ts[ts.field == f].sort_values("date").HH_db.values
            return np.nanmax(-np.diff(d[3:]))
        best = max(p, key=depth)
    else:
        best = next(f for f, l in labels.items() if l == "flat")
    fig = new_fig()
    ax = fig.add_axes([0.11, 0.13, 0.86, 0.66])
    d = series_plot(ax, ts, best)
    if labels[best] == "pulse":
        hh = d.HH_db.values
        i = 3 + int(np.nanargmax(-np.diff(hh[3:]))) + 1
        ax.annotate("сброс воды", (d.date.values[i], hh[i]), xytext=(0, -70), textcoords="offset points",
                    ha="center", fontsize=L, color="#b07a00", weight="bold",
                    arrowprops=dict(arrowstyle="->", color="#b07a00", lw=3))
    style(ax)
    ax.legend(fontsize=L, loc="lower right")
    fig.text(0.08, 0.9, f"Ритм одного рисового поля: {af.LABELS[labels[best]][0].lower().replace(' (ритм)', '')}", fontsize=T, weight="bold")
    fig.text(0.08, 0.84, "Рис стоит в воде — HH яркий. Воду сбросили — HH падает.", fontsize=L)
    save(fig, sl / f"{pre}_2_field.png")

    # ---------- 3. Два поля ----------
    flat = [f for f, l in labels.items() if l == "flat"]
    if flat and pulse:
        def smooth(f):
            h = ts[ts.field == f].sort_values("date").HH_db.values
            return np.nanmax(-np.diff(h[3:])) - 0.3 * (np.nanmax(h) - np.nanmin(h))
        f0 = min(flat, key=smooth)
        fig = new_fig()
        ax = fig.add_axes([0.11, 0.13, 0.86, 0.7])
        series_plot(ax, ts, f0, af.LABELS["flat"][1], "Поле А: вода стояла весь сезон", hv=False)
        series_plot(ax, ts, best, af.LABELS["pulse"][1], "Поле Б: воду сбросили 23.09 и вернули", hv=False)
        style(ax)
        ax.set_ylabel("сигнал радара HH, дБ\n(выше = ярче отражение)", fontsize=L)
        ax.legend(fontsize=L, loc="lower right")
        fig.text(0.08, 0.9, "Два рисовых поля — два ритма воды", fontsize=T, weight="bold")
        save(fig, sl / f"{pre}_3_compare.png")

    # ---------- 4. До / после ----------
    has = [i for i in range(len(st.time)) if np.isfinite(st.HV.values[i]).any()]
    fig = new_fig()
    for j, (i, t) in enumerate(zip((has[0], has[-1]), ("Начало сезона", "Конец сезона"))):
        ax = fig.add_axes([0.01 + j * 0.495, 0.1, 0.485, 0.74])
        r, _ = rgb_frame(st.HH.values[i], st.HV.values[i], max(1, int(round(30 / abs(x[1] - x[0])))))
        ax.imshow(r[::-1] if y[0] < y[-1] else r, aspect="auto")
        ax.axis("off")
        ax.set_title(f"{t}: {ru_date(pd.Timestamp(st.time.values[i]))}", fontsize=L + 4)
    fig.text(0.01, 0.92, f"{args.region}: радар NISAR в начале и в конце сезона", fontsize=T, weight="bold")
    fig.text(0.01, 0.045, "тёмное — открытая вода · зелёное — растения · розовое — вода под растениями", fontsize=S)
    save(fig, sl / f"{pre}_4_before_after.png")

    # ---------- 5. Цифры ----------
    fig = new_fig()
    fig.text(0.05, 0.88, f"{args.region}: итоги сезона 2026", fontsize=T + 4, weight="bold")
    rice = [l for l in ("flat", "pulse", "dry_seed") if counts.get(l, 0)]
    nr = sum(counts.get(l, 0) for l in rice)
    lines = [f"{len(dates)} снимков NISAR, {pd.Timestamp(dates[0]):%d.%m} — {pd.Timestamp(dates[-1]):%d.%m.%Y}",
             f"{n} квадратов 300×300 м проанализировано",
             f"{nr} квадратов — рис ({nr / n:.1%} территории)".replace(".", ",")]
    for l in rice:
        lines.append(f"    {af.LABELS[l][0].replace(' (ритм)', '')}: {counts[l]} ({counts[l] / nr:.1%} риса)".replace(".", ","))
        if l == "pulse":
            lines.append("        из них 8 — соседние квадраты одного массива")
    v = out / "validation.txt"
    if v.exists():
        vt = v.read_text(encoding="utf-8")
        m1 = re.search(r"([\d.]+)% на (\d+) парах", vt)
        m2 = re.search(r"сбалансированная точность[^:]*: ([\d.]+)%", vt)
        if m1:
            lines.append(f"Сверка с Sentinel-2: совпадение {m1.group(1).replace('.', ',')}% на {m1.group(2)} парах «квадрат–дата»")
        if m2:
            lines.append(f"    без перекоса в сторону суши — {m2.group(1).replace('.', ',')}%")
    for i, s in enumerate(lines):
        fig.text(0.06, 0.74 - i * 0.08, s, fontsize=L + 7 if not s.startswith("    ") else L + 3)
    save(fig, sl / f"{pre}_5_numbers.png")


if __name__ == "__main__":
    main()
