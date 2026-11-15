#!/usr/bin/env python3
"""Кадры по датам + season.gif + before_after.png (с подписью источника) из stack.nc.

  python make_visuals.py --stack data/stack.nc --region "Жалагаш, Кызылорда"
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

CREDIT = "NISAR data courtesy of NASA/JPL, ASF DAAC"
MONTHS = ["", "января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
          "сентября", "октября", "ноября", "декабря"]


def ru_date(d):
    return f"{d.day} {MONTHS[d.month]} {d.year}"


def to_db(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 10 * np.log10(a)
    return np.where(np.isfinite(d), d, np.nan)


def block_mean(a, k):
    """Усреднение k×k — убирает спекл-шум радара для картинки."""
    ny, nx = a.shape[0] // k * k, a.shape[1] // k * k
    a = a[:ny, :nx].reshape(ny // k, k, nx // k, k)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmean(a, axis=(1, 3))


def rgb_frame(hh, hv, k):
    hh = to_db(block_mean(np.where(hh > 0, hh, np.nan), k))
    hv = to_db(block_mean(np.where(hv > 0, hv, np.nan), k)) if np.isfinite(hv).any() else None
    n = lambda a, lo, hi: np.clip((a - lo) / (hi - lo), 0, 1)
    g = n(hv, -30, -10) if hv is not None else n(hh, -25, 0) * 0.6
    rgb = np.dstack([n(hh, -22, -2), g, n(hh, -22, -2)])
    return np.nan_to_num(rgb), hv is None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="data/stack.nc")
    ap.add_argument("--region", default="")
    ap.add_argument("--out", default="out")
    ap.add_argument("--k", type=int, default=0, help="усреднение пикселей (0 = авто)")
    args = ap.parse_args()

    st = xr.open_dataset(args.stack, engine="h5netcdf").load()
    data_dir = Path(args.stack).parent
    dates = pd.to_datetime(st.time.values)
    flip = st.y.values[0] < st.y.values[-1]
    k = args.k or max(1, int(round(30 / float(abs(st.x.values[1] - st.x.values[0])))))
    track = st.attrs.get("track", "")

    frames = []
    qdir = data_dir / "quicklooks"
    qdir.mkdir(exist_ok=True)
    for i, d in enumerate(dates):
        rgb, no_hv = rgb_frame(st.HH.values[i], st.HV.values[i], k)
        if flip:
            rgb = rgb[::-1]
        fig = plt.figure(figsize=(8, 7.2), dpi=110)
        ax = fig.add_axes([0.02, 0.07, 0.96, 0.85])
        ax.imshow(rgb)
        ax.axis("off")
        fig.suptitle(f"{args.region} · NISAR L-band · {ru_date(d)}", fontsize=13, y=0.975)
        sub = "розовое = HH (вода с растениями, почва) · зелёное = HV (растения) · тёмное = открытая вода"
        if no_hv:
            sub = "в эту дату только HH (без HV)"
        fig.text(0.5, 0.935, sub, ha="center", fontsize=8.5, color="#333")
        fig.text(0.99, 0.015, CREDIT, ha="right", fontsize=8, color="#555")
        fig.text(0.01, 0.015, f"трек {track}", ha="left", fontsize=8, color="#555")
        p = qdir / f"{d:%Y%m%d}.png"
        fig.savefig(p)
        plt.close(fig)
        frames.append(p)

    import imageio.v2 as imageio
    imgs = [imageio.imread(p) for p in frames]
    h = min(i.shape[0] for i in imgs)
    w = min(i.shape[1] for i in imgs)
    imageio.mimsave(data_dir / "season.gif", [i[:h, :w] for i in imgs], duration=1000, loop=0)
    print(f"GIF: {data_dir / 'season.gif'} ({len(frames)} кадров)")

    # До / после: первая и последняя дата с HV
    has_hv = [i for i in range(len(dates)) if np.isfinite(st.HV.values[i]).any()]
    a, b = has_hv[0], has_hv[-1]
    fig, axs = plt.subplots(1, 2, figsize=(14, 7), dpi=130)
    for ax, i, t in zip(axs, (a, b), ("Начало сезона", "Конец сезона")):
        rgb, _ = rgb_frame(st.HH.values[i], st.HV.values[i], k)
        if flip:
            rgb = rgb[::-1]
        ax.imshow(rgb)
        ax.set_title(f"{t}: {ru_date(dates[i])}", fontsize=15)
        ax.axis("off")
    fig.suptitle(f"{args.region}: радар NISAR в начале и в конце сезона", fontsize=17)
    fig.text(0.5, 0.04, "розовое = HH (вода и почва) · зелёное = HV (растения) · тёмное = открытая вода",
             ha="center", fontsize=11)
    fig.text(0.99, 0.01, CREDIT, ha="right", fontsize=10, color="#555")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.08, wspace=0.03)
    Path(args.out).mkdir(exist_ok=True)
    out = Path(args.out) / "before_after.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"До/после: {out}")


if __name__ == "__main__":
    main()
