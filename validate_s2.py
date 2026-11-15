#!/usr/bin/env python3
"""Сверка радара с оптикой Sentinel-2 L2A (Microsoft Planetary Computer, без авторизации).

Для каждой даты NISAR ищем безоблачный (облачность сцены < 20%) снимок Sentinel-2 в пределах ±3 дней.
По каждому квадрату: доля пикселей воды по MNDWI = (B03 − B11) / (B03 + B11) > 0 (облака по SCL исключены).
Оптика «вода» = больше половины чистых пикселей квадрата — вода.
Радар «вода» = HH < WATER_HH_DB (то же правило, что в analyze_fields.py).
Сравниваем только сельхоз-квадраты (рис + другие культуры), иначе пустыня завысит совпадение.

  python validate_s2.py --stack data/stack.nc --out out
"""
import argparse
import json
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
CREDIT = "NISAR data courtesy of NASA/JPL, ASF DAAC · Sentinel-2: Copernicus/ESA via Microsoft Planetary Computer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="data/stack.nc")
    ap.add_argument("--out", default="out")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--cloud", type=float, default=20)
    ap.add_argument("--reuse", action="store_true", help="не качать Sentinel-2, пересчитать по validation.csv")
    args = ap.parse_args()
    out = Path(args.out)
    if args.reuse:
        df = pd.read_csv(out / "validation.csv")
        df["nisar_date"] = pd.to_datetime(df.nisar_date).dt.date
        return report(df, out)

    import analyze_fields as af
    import pystac_client, planetary_computer, rasterio
    from rasterio.features import rasterize
    from rasterio.warp import transform_geom, transform_bounds
    from rasterio.windows import from_bounds
    from shapely.geometry import shape

    gj = json.loads((out / "fields_result.geojson").read_text())
    feats = [f for f in gj["features"] if f["properties"]["label"] in ("flat", "pulse", "dry_seed", "other_crop")]
    ids = [f["properties"]["id"] for f in feats]
    print(f"сельхоз-квадратов: {len(ids)}")
    ts = pd.read_csv(out / "timeseries.csv", parse_dates=["date"])
    ts = ts[ts.field.isin(ids)]
    nisar_dates = sorted(ts.date.unique())

    allg = [shape(f["geometry"]) for f in feats]
    minx = min(g.bounds[0] for g in allg); miny = min(g.bounds[1] for g in allg)
    maxx = max(g.bounds[2] for g in allg); maxy = max(g.bounds[3] for g in allg)

    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)
    t0 = pd.Timestamp(nisar_dates[0]) - timedelta(days=args.days)
    t1 = pd.Timestamp(nisar_dates[-1]) + timedelta(days=args.days)
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=[minx, miny, maxx, maxy],
                            datetime=f"{t0:%Y-%m-%d}/{t1:%Y-%m-%d}",
                            query={"eo:cloud_cover": {"lt": args.cloud}}).items())
    print(f"сцен Sentinel-2 с облачностью < {args.cloud}%: {len(items)}")

    pairs = []  # (nisar_date, s2_item)
    for d in nisar_dates:
        d = pd.Timestamp(d)
        near = [it for it in items if abs((pd.Timestamp(it.datetime).tz_localize(None) - d).days) <= args.days]
        if near:
            # ближайшая по времени, потом наименее облачная
            near.sort(key=lambda it: (abs((pd.Timestamp(it.datetime).tz_localize(None) - d).total_seconds()),
                                      it.properties["eo:cloud_cover"]))
            # все сцены этой же даты (участок может лежать на двух тайлах)
            day = pd.Timestamp(near[0].datetime).date()
            pairs.append((d, [it for it in near if pd.Timestamp(it.datetime).date() == day]))
    print(f"дат NISAR с парой Sentinel-2 (±{args.days} дн.): {len(pairs)} из {len(nisar_dates)}")

    rows = []
    for d, its in pairs:
        water = {}   # id -> [water_px, clear_px]
        for it in its:
            with rasterio.open(it.assets["B11"].href) as src:
                crs = src.crs
                b = transform_bounds("EPSG:4326", crs, minx, miny, maxx, maxy)
                win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
                win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                swir = src.read(1, window=win).astype("float32")
                tf = src.window_transform(win)
            shp = swir.shape
            with rasterio.open(it.assets["B03"].href) as src:
                b10 = transform_bounds("EPSG:4326", src.crs, minx, miny, maxx, maxy)
                w10 = from_bounds(*b, transform=src.transform)
                green = src.read(1, window=w10, out_shape=shp, boundless=True, fill_value=0).astype("float32")
            with rasterio.open(it.assets["SCL"].href) as src:
                scl = src.read(1, window=from_bounds(*b, transform=src.transform), out_shape=shp,
                               boundless=True, fill_value=0)
            clear = np.isin(scl, [4, 5, 6, 7, 11]) & (green + swir > 0)  # растения, почва, вода, неклассиф., снег; без облаков/теней
            mndwi = (green - swir) / np.maximum(green + swir, 1)
            shapes = [(transform_geom("EPSG:4326", crs, f["geometry"]), k + 1) for k, f in enumerate(feats)]
            lab = rasterize(shapes, out_shape=shp, transform=tf, fill=0, dtype="int32")
            n = len(feats) + 1
            cw = np.bincount(lab[clear & (mndwi > 0)].ravel(), minlength=n)
            cc = np.bincount(lab[clear].ravel(), minlength=n)
            tot = np.bincount(lab.ravel(), minlength=n)
            for k in range(1, n):
                if tot[k] == 0:
                    continue
                w = water.setdefault(ids[k - 1], [0, 0, 0])
                w[0] += cw[k]; w[1] += cc[k]; w[2] += tot[k]
        s2date = pd.Timestamp(its[0].datetime).tz_localize(None)
        sub = ts[ts.date == d].set_index("field")
        for fid, (wpx, cpx, tpx) in water.items():
            if cpx < 0.8 * tpx or fid not in sub.index:   # квадрат на 20%+ в облаках/вне снимка
                continue
            hh = sub.loc[fid, "HH_db"]
            if not np.isfinite(hh):
                continue
            rows.append({"field": fid, "nisar_date": d.date(), "s2_date": s2date.date(),
                         "dt_days": (s2date - d).days, "HH_db": hh, "water_frac_s2": wpx / cpx,
                         "radar_water": bool(hh < af.WATER_HH_DB), "optic_water": bool(wpx / cpx > 0.5)})
        print(f"  {d:%d.%m} ↔ S2 {s2date:%d.%m} ({len(its)} тайл.): пар {sum(r['nisar_date'] == d.date() for r in rows)}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "validation.csv", index=False)
    report(df, out)


def report(df, out):
    import analyze_fields as af
    if df.empty:
        print("пар нет — сверка не посчитана")
        return
    agree = (df.radar_water == df.optic_water).mean()
    tp = int((df.radar_water & df.optic_water).sum()); tn = int((~df.radar_water & ~df.optic_water).sum())
    fp = int((df.radar_water & ~df.optic_water).sum()); fn = int((~df.radar_water & df.optic_water).sum())
    # «сбалансированная» точность: среднее из совпадения по воде и по суше (не завышается сушей)
    bal = np.nanmean([tp / (tp + fn) if tp + fn else np.nan, tn / (tn + fp) if tn + fp else np.nan])
    txt = (f"Совпадение радар/оптика «вода — нет воды»: {agree:.1%} на {len(df)} парах поле-дата "
           f"({df.nisar_date.nunique()} дат, {df.field.nunique()} квадратов)\n"
           f"  обе «вода»: {tp}, обе «нет»: {tn}, радар вода/оптика нет: {fp}, радар нет/оптика вода: {fn}\n"
           f"  сбалансированная точность (без перекоса в сторону суши): {bal:.1%}")
    nd = pd.to_datetime(df.nisar_date)
    for name, sel in [(f"начало сезона (до {af.EARLY_END[1]:02d}.{af.EARLY_END[0]:02d}, только эти даты используются правилом «весенняя вода»)",
                       [(d.month, d.day) <= af.EARLY_END for d in nd]),
                      ("середина сезона (16.07–15.09, рис закрывает воду)", [(7, 16) <= (d.month, d.day) <= (9, 15) for d in nd]),
                      ("конец сезона (после 15.09, уборка)", [(d.month, d.day) > (9, 15) for d in nd])]:
        s_ = df[np.array(sel)]
        if len(s_):
            txt += (f"\n  {name}: совпадение {(s_.radar_water == s_.optic_water).mean():.1%} на {len(s_)} парах; "
                    f"радар «вода» {int(s_.radar_water.sum())}, оптика «вода» {int(s_.optic_water.sum())}, "
                    f"обе {int((s_.radar_water & s_.optic_water).sum())}")
    print(txt)
    (out / "validation.txt").write_text(txt + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), dpi=120, gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axs[0]
    c = np.where(df.radar_water == df.optic_water, "#2e7d32", "#d64545")
    ax.scatter(df.HH_db, df.water_frac_s2, s=6, c=c, alpha=0.5)
    ax.axvline(af.WATER_HH_DB, color="#333", ls="--", lw=1)
    ax.axhline(0.5, color="#333", ls="--", lw=1)
    ax.set_xlabel("Радар NISAR: HH, дБ  (левее линии = «вода»)")
    ax.set_ylabel("Sentinel-2: доля воды в квадрате (MNDWI > 0)")
    ax.set_title("Каждая точка — квадрат 300 м в одну дату\nзелёные — радар и оптика совпали, красные — нет")
    ax = axs[1]
    m = np.array([[tp, fn], [fp, tn]])
    ax.imshow(m, cmap="Greens")
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=16, color="white" if v > m.max() / 2 else "black")
    ax.set_xticks([0, 1], ["радар: вода", "радар: нет"])
    ax.set_yticks([0, 1], ["оптика: вода", "оптика: нет"])
    ax.set_title(f"Совпадение {agree:.1%} на {len(df)} парах\n(сбаланс. {bal:.1%})")
    fig.text(0.995, 0.005, CREDIT, ha="right", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "validation.png")
    print(f"→ {out / 'validation.png'}")


if __name__ == "__main__":
    main()
