#!/usr/bin/env python3
"""
Rice Rhythm — шаг 1: забрать данные NISAR GCOV по нашему участку.

Что делает:
  1. Ищет все снимки NISAR GCOV над полигоном за сезон (asf_search).
  2. Оставляет только один трек (по умолчанию 078, нисходящий).
  3. Для каждого снимка читает по сети ТОЛЬКО нужный кусок слоёв HH и HV
     (не качает 2 ГБ целиком).
  4. Сохраняет всё в один компактный файл stack.nc
     + картинки-превью по датам (для слайдов и GIF).

Запуск:
  export EARTHDATA_TOKEN="ваш_токен"
  python fetch_nisar.py

Если потоковое чтение не заработает, запустите с флагом --download:
  python fetch_nisar.py --download
  (тогда файлы скачаются целиком в папку raw/, ~2 ГБ каждый)
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# НАСТРОЙКИ — меняйте здесь
# ---------------------------------------------------------------------------
# Участок: долгота_мин, широта_мин, долгота_макс, широта_макс (градусы)
AOI_BOUNDS = (64.35, 44.95, 64.65, 45.12)          # Жалагаш, Кызылорда
# Для второго региона (Карнал, Харьяна) раскомментируйте:
# AOI_BOUNDS = (76.88, 29.60, 77.09, 29.78)

START = datetime(2026, 6, 17)
END = datetime(2026, 10, 31, 23, 59)
TRACK = 78            # номер трека (в названии файла _078_)
DIRECTION = "D"       # D — нисходящий, A — восходящий
OUT_DIR = Path("data")
# ---------------------------------------------------------------------------

GRID = "/science/LSAR/GCOV/grids/frequencyA"


def aoi_wkt(b):
    x0, y0, x1, y1 = b
    return f"POLYGON(({x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}))"


def parse_name(url):
    """NISAR_L2_PR_GCOV_034_078_D_066_2005_DHDH_A_20261029T145013_..._P05036_N_F_J_001.h5"""
    name = os.path.basename(url)
    parts = name.replace(".h5", "").split("_")
    try:
        info = {
            "name": name,
            "cycle": int(parts[4]),
            "track": int(parts[5]),
            "direction": parts[6],
            "frame": int(parts[7]),
            "pol": parts[9],
            "start": datetime.strptime(parts[11], "%Y%m%dT%H%M%S"),
            "crid": next((p for p in parts if re.fullmatch(r"[PX]\d{5}", p)), "P00000"),
        }
    except (IndexError, ValueError):
        return None
    return info


REGIONS = {
    "kzyl": dict(aoi=(64.35, 44.95, 64.65, 45.12), track=78, direction="D", out="data"),
    "karnal": dict(aoi=(76.88, 29.60, 77.09, 29.78), track=None, direction=None, out="data_karnal"),
    "arkansas": dict(aoi=(-91.763, 34.578, -91.740, 34.596), track=None, direction=None, out="data_arkansas"),
}


def search(session, all_tracks=False):
    import asf_search as asf

    opts = {
        "dataset": ["NISAR"],
        "processingLevel": ["GCOV"],
        "productionConfiguration": ["PR"],
        "intersectsWith": aoi_wkt(AOI_BOUNDS),
        "start": START,
        "end": END,
        "maxResults": 500,
    }
    results = asf.search(**opts)
    print(f"Найдено продуктов GCOV над участком: {len(results)}")

    urls = []
    try:
        urls = list(results.find_urls(extension=".h5", directAccess=False))
    except Exception:
        for p in results:
            for u in [p.properties.get("url")] + list(p.properties.get("additionalUrls") or []):
                if u and u.endswith(".h5"):
                    urls.append(u)
    urls = sorted(set(urls))

    if all_tracks:
        dates = {}
        for u in urls:
            i = parse_name(u)
            if i:
                dates.setdefault((i["track"], i["direction"]), set()).add(i["start"].date())
        for k, v in sorted(dates.items(), key=lambda kv: -len(kv[1])):
            print(f"  трек {k[0]:03d}{k[1]}: {len(v)} дат: " + ", ".join(f"{d:%d.%m}" for d in sorted(v)))
        return dates
    # Оставляем нужный трек и направление
    items = []
    for u in urls:
        info = parse_name(u)
        if info and info["track"] == TRACK and info["direction"] == DIRECTION:
            info["url"] = u
            items.append(info)

    # Убираем дубли версий: для одной даты+фрейма берём самую новую версию (CRID)
    best = {}
    for it in items:
        key = (it["start"].date(), it["frame"])
        if key not in best or it["crid"] > best[key]["crid"]:
            best[key] = it
    items = list(best.values())

    # Если на одну дату приходится несколько фреймов — берём самый частый фрейм
    if items:
        frames = [it["frame"] for it in items]
        main_frame = max(set(frames), key=frames.count)
        items = [it for it in items if it["frame"] == main_frame]

    items.sort(key=lambda it: it["start"])
    print(f"Трек {TRACK:03d}{DIRECTION}: {len(items)} дат")
    for it in items:
        print(f"  {it['start']:%Y-%m-%d}  фрейм {it['frame']:03d}  {it['crid']}  {it['pol']}")
    return items


def read_epsg(grid):
    proj = grid["projection"]
    for key in ("epsg_code", "epsg"):
        if key in proj.attrs:
            return int(np.array(proj.attrs[key]).item())
    return int(np.array(proj[()]).item())


def read_subset(h5file):
    """Читает кусок HH и HV по AOI. Возвращает dict с массивами."""
    import h5py
    from pyproj import Transformer

    with h5py.File(h5file, "r") as f:
        grid = f[GRID]
        epsg = read_epsg(grid)
        x = grid["xCoordinates"][:]
        y = grid["yCoordinates"][:]

        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        x0, y0, x1, y1 = AOI_BOUNDS
        xs, ys = tr.transform([x0, x1, x1, x0], [y0, y0, y1, y1])
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)

        ix = np.where((x >= minx) & (x <= maxx))[0]
        iy = np.where((y >= miny) & (y <= maxy))[0]
        if ix.size == 0 or iy.size == 0:
            raise ValueError("Участок не попал в снимок")
        xs_sl = slice(ix.min(), ix.max() + 1)
        ys_sl = slice(iy.min(), iy.max() + 1)

        out = {"epsg": epsg, "x": x[xs_sl], "y": y[ys_sl]}
        for pol in ("HHHH", "HVHV"):
            if pol in grid:
                out[pol] = grid[pol][ys_sl, xs_sl].astype("float32")
            else:
                print(f"    ! слоя {pol} нет в файле")
        return out


def open_stream(url, token):
    import fsspec

    fs = fsspec.filesystem(
        "https",
        headers={"Authorization": f"Bearer {token}"},
        block_size=4 * 1024 * 1024,
    )
    return fs.open(url, "rb", cache_type="blockcache")


def to_db(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 10 * np.log10(a)
    d[~np.isfinite(d)] = np.nan
    return d


def save_quicklooks(stack_hh, stack_hv, dates, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qdir = out / "quicklooks"
    qdir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, d in enumerate(dates):
        hh = to_db(stack_hh[i])
        hv = to_db(stack_hv[i]) if stack_hv is not None else hh
        # Ложные цвета: R = HH, G = HV, B = HH (розовое = HH, зелёное = HV)
        def norm(a, lo, hi):
            return np.clip((a - lo) / (hi - lo), 0, 1)
        rgb = np.dstack([norm(hh, -25, 0), norm(hv, -32, -8), norm(hh, -25, 0)])
        rgb = np.nan_to_num(rgb)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
        ax.imshow(rgb)
        ax.set_title(f"NISAR L-band · {d:%d.%m.%Y} · розовое = HH (вода/почва), зелёное = HV (растения)",
                     fontsize=9)
        ax.axis("off")
        p = qdir / f"{d:%Y%m%d}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        frames.append(p)
    try:
        import imageio.v2 as imageio
        imgs = [imageio.imread(p) for p in frames]
        h = min(i.shape[0] for i in imgs)
        w = min(i.shape[1] for i in imgs)
        imgs = [i[:h, :w] for i in imgs]
        imageio.mimsave(out / "season.gif", imgs, duration=1.0, loop=0)
        print(f"GIF сезона: {out / 'season.gif'}")
    except Exception as e:
        print(f"GIF не собран ({e}), но PNG по датам есть в {qdir}")


def main():
    global AOI_BOUNDS, TRACK, DIRECTION, OUT_DIR, START, END
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true",
                    help="скачать файлы целиком вместо потокового чтения")
    ap.add_argument("--list-only", action="store_true",
                    help="только показать список дат и выйти")
    ap.add_argument("--region", default="kzyl", choices=list(REGIONS))
    ap.add_argument("--track", type=int)
    ap.add_argument("--direction")
    ap.add_argument("--tracks", action="store_true", help="показать все треки над участком")
    ap.add_argument("--out", help="папка для результатов (по умолчанию из REGIONS)")
    ap.add_argument("--start", help="YYYY-MM-DD, начало окна дат")
    ap.add_argument("--end", help="YYYY-MM-DD, конец окна дат")
    args = ap.parse_args()
    if args.start: START = datetime.fromisoformat(args.start)
    if args.end: END = datetime.fromisoformat(args.end + "T23:59")
    r = REGIONS[args.region]
    AOI_BOUNDS = r["aoi"]
    TRACK = args.track if args.track is not None else r["track"]
    DIRECTION = args.direction or r["direction"]
    OUT_DIR = Path(args.out or r["out"])
    if args.tracks:
        search(None, all_tracks=True)
        return
    if TRACK is None or DIRECTION is None:
        sys.exit("Укажите --track и --direction (см. --tracks)")

    import asf_search as asf

    token = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if not token and not args.list_only:
        sys.exit("Нет токена. Выполните: export EARTHDATA_TOKEN=\"ваш_токен\"")

    session = asf.ASFSession()
    if token:
        session.auth_with_token(token)

    items = search(session)
    if not items:
        sys.exit("Снимков не найдено — проверьте AOI_BOUNDS, TRACK, DIRECTION, даты.")
    if args.list_only:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = OUT_DIR / "raw"

    hh_list, hv_list, dates = [], [], []
    ref = None
    for it in items:
        print(f"→ {it['start']:%Y-%m-%d}: читаю кусок...", flush=True)
        try:
            if args.download:
                raw.mkdir(parents=True, exist_ok=True)
                local = raw / it["name"]
                if not local.exists():
                    asf.download_urls([it["url"]], str(raw), session=session)
                sub = read_subset(local)
            else:
                with open_stream(it["url"], token) as fobj:
                    sub = read_subset(fobj)
        except Exception as e:
            print(f"    ! пропуск: {e}")
            continue

        if ref is None:
            ref = sub
        # Приводим к общей сетке (у одного трека сетка совпадает; на всякий случай обрезаем)
        ny = min(ref["HHHH"].shape[0], sub["HHHH"].shape[0])
        nx = min(ref["HHHH"].shape[1], sub["HHHH"].shape[1])
        hh_list.append(sub["HHHH"][:ny, :nx])
        hv_list.append(sub.get("HVHV", np.full_like(sub["HHHH"], np.nan))[:ny, :nx])
        dates.append(it["start"])
        print(f"    ок: {ny}×{nx} пикселей")

    if not dates:
        sys.exit("Ни один снимок не прочитался. Попробуйте флаг --download.")

    ny = min(a.shape[0] for a in hh_list)
    nx = min(a.shape[1] for a in hh_list)
    hh = np.stack([a[:ny, :nx] for a in hh_list])
    hv = np.stack([a[:ny, :nx] for a in hv_list])

    import xarray as xr
    ds = xr.Dataset(
        {
            "HH": (("time", "y", "x"), hh),
            "HV": (("time", "y", "x"), hv),
        },
        coords={
            "time": np.array(dates, dtype="datetime64[ns]"),
            "y": ref["y"][:ny],
            "x": ref["x"][:nx],
        },
        attrs={
            "epsg": ref["epsg"],
            "source": "NASA-ISRO NISAR L2 GCOV, ASF DAAC",
            "units": "gamma0 linear power",
            "track": f"{TRACK:03d}{DIRECTION}",
        },
    )
    path = OUT_DIR / "stack.nc"
    ds.to_netcdf(path, engine="h5netcdf")
    print(f"\nГотово: {path}  ({len(dates)} дат, {ny}×{nx} пикселей, EPSG:{ref['epsg']})")

    save_quicklooks(hh, hv, dates, OUT_DIR)


if __name__ == "__main__":
    main()
