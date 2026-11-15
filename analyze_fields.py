#!/usr/bin/env python3
"""
Rice Rhythm — шаг 2: ритм каждого поля, классификация, карта.

Вход:
  data/stack.nc        — из fetch_nisar.py
  fields.geojson       — контуры полей (нарисовать на geojson.io, сохранить сюда)
                         Если файла нет — участок режется на квадраты 300×300 м (план Б).

Выход (папка out/):
  timeseries.csv       — значения HH и HV (в дБ) по каждому полю и дате
  fields_result.geojson— поля с меткой режима
  charts/<id>.png      — график ритма каждого поля
  compare.png          — два поля рядом: «ровное» и «пульсирующее» (для слайда)
  map.html             — интерактивная карта (открыть в браузере → скриншоты)
  summary.txt          — итоговые цифры для слайда
  sound/<id>.wav       — звук ритма (если запустить с --sound)

Запуск:
  python analyze_fields.py
  python analyze_fields.py --sound
"""

import argparse
import base64
import json
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# ПОРОГИ — подбираются по графикам. Начальные значения — разумная догадка.
# ---------------------------------------------------------------------------
DROP_DB = None       # падение HH (дБ) = «сброс воды». None = авто: 99-й процентиль шума
RISE_DB = None       # рост HH обратно (дБ) = «повторный залив». None = авто: 95-й процентиль шума
                     # шум = |ΔHH| между соседними датами у НЕрисовых квадратов после вычета
                     # общего для участка сдвига (погода/влажность)
RICE_HV_GROWTH = 3.0 # рис: HV после 15.07 поднимается над минимумом начала сезона хотя бы на столько
                     # (провал гистограммы роста HV по Кызылорде — между 2.5 и 3.0 дБ)
HARVEST_HV_DROP = 3.0  # падение HV (дБ) к концу сезона = уборка
NOT_RICE_HV_MAX = -22.0  # если максимум HV за сезон ниже — мало растительности, вероятно не рис
WATER_HH_DB = -16.5  # HH ниже — открытая вода. Рисовые квадраты: мода весеннего минимума HH −20, у нериса −15; провал гистограммы у риса −17…−16 (квадрат 300 м смешивает чек и дамбы)
EARLY_END = (7, 15)  # «начало сезона» — даты до 15 июля включительно
DRY_SEED_MIN_FRAC = 0.25  # правило прямого посева включается, если среди рисовых полей
                          # хотя бы такая доля затоплена в начале сезона
GRID_CELL_M = 300    # размер квадрата для плана Б, метры
MAX_CHARTS_PER_LABEL = 40  # графиков (и попапов/звуков) на категорию — чтобы карта не весила 100 МБ
CREDIT = "NISAR data courtesy of NASA/JPL, ASF DAAC"
MIN_PIXELS = 6       # минимум пикселей в поле, иначе поле пропускаем
# ---------------------------------------------------------------------------

LABELS = {
    "pulse": ("Воду сбрасывали (ритм)", "#f2a900"),
    "flat": ("Вода стояла весь сезон", "#1f6fd1"),
    "dry_seed": ("Рис без затопления в начале (прямой посев)", "#2e9e44"),
    "other_crop": ("Другие культуры (растут без затопления)", "#8bc34a"),
    "not_rice": ("Не рис / мало растительности", "#9a9a9a"),
    "unclear": ("Неясно", "#d64545"),
}


def add_credit(fig, size=7):
    fig.text(0.995, 0.005, CREDIT, ha="right", va="bottom", fontsize=size, color="#555")


def grid_series(stack):
    """План Б быстро: весь участок режется на квадраты GRID_CELL_M, медиана по пикселям."""
    from shapely.geometry import Polygon
    from pyproj import Transformer
    epsg = int(stack.attrs["epsg"])
    to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    x, y = stack.x.values, stack.y.values
    dx, dy = float(abs(x[1] - x[0])), float(abs(y[1] - y[0]))
    k = max(1, int(round(GRID_CELL_M / dx)))
    nby, nbx = len(y) // k, len(x) // k
    out = {}
    for var in ("HH", "HV"):
        a = stack[var].values[:, :nby * k, :nbx * k].astype("float32")
        a = np.where(a > 0, a, np.nan)
        a = a.reshape(a.shape[0], nby, k, nbx, k).transpose(0, 1, 3, 2, 4).reshape(a.shape[0], nby, nbx, k * k)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[var] = to_db(np.nanmedian(a, axis=-1))
            if var == "HH":
                valid = np.isfinite(a).sum(-1).min(0)
    cells = []
    for j in range(nby):
        for i in range(nbx):
            if valid[j, i] < MIN_PIXELS:
                continue
            xa, xb = x[i * k] - dx / 2, x[(i + 1) * k - 1] + dx / 2
            ya, yb = y[j * k], y[(j + 1) * k - 1]
            ya, yb = min(ya, yb) - dy / 2, max(ya, yb) + dy / 2
            lons, lats = to_ll.transform([xa, xb, xb, xa], [ya, ya, yb, yb])
            cells.append((f"C{j:03d}_{i:03d}", Polygon(zip(lons, lats)),
                          out["HH"][:, j, i], out["HV"][:, j, i], int(valid[j, i])))
    print(f"Сетка {GRID_CELL_M} м: {nby}×{nbx}, квадратов с данными: {len(cells)} (пиксель {dx:.0f} м)")
    return cells


def to_db(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 10 * np.log10(a)
    return np.where(np.isfinite(d), d, np.nan)


def load_fields(path, stack):
    """Возвращает список (id, shapely-геометрия в EPSG:4326)."""
    from shapely.geometry import shape, box
    from pyproj import Transformer

    if path.exists():
        gj = json.loads(path.read_text())
        out = []
        for i, feat in enumerate(gj["features"]):
            fid = str(feat.get("properties", {}).get("name") or f"F{i + 1:03d}")
            out.append((fid, shape(feat["geometry"])))
        print(f"Полей из {path}: {len(out)}")
        return out

    print(f"! {path} не найден — режу участок на квадраты {GRID_CELL_M} м (план Б)")
    epsg = int(stack.attrs["epsg"])
    to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    x, y = stack.x.values, stack.y.values
    side = int(np.sqrt(GRID_MAX_CELLS))
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    half = side * GRID_CELL_M / 2
    xs = np.arange(max(x.min(), cx - half), min(x.max(), cx + half) - GRID_CELL_M + 1, GRID_CELL_M)
    ys = np.arange(max(y.min(), cy - half), min(y.max(), cy + half) - GRID_CELL_M + 1, GRID_CELL_M)
    out = []
    for i, x0 in enumerate(xs):
        for j, y0 in enumerate(ys):
            (lx0, lx1), (ly0, ly1) = to_ll.transform([x0, x0 + GRID_CELL_M], [y0, y0 + GRID_CELL_M])
            out.append((f"C{i:02d}_{j:02d}", box(lx0, ly0, lx1, ly1)))
    return out


def field_series(stack, geom_ll):
    """Медиана HH и HV (в линейных единицах → дБ) по пикселям поля для каждой даты."""
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin

    epsg = int(stack.attrs["epsg"])
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform
    geom = shp_transform(to_utm, geom_ll)

    x, y = stack.x.values, stack.y.values
    dx = float(abs(x[1] - x[0]))
    dy = float(abs(y[1] - y[0]))
    flip = y[0] < y[-1]  # строки должны идти сверху вниз
    top = float(max(y[0], y[-1])) + dy / 2
    left = float(x[0]) - dx / 2
    tf = from_origin(left, top, dx, dy)
    mask = geometry_mask([geom.__geo_interface__], out_shape=(len(y), len(x)),
                         transform=tf, invert=True)
    if flip:
        mask = mask[::-1, :]
    n = int(mask.sum())
    if n < MIN_PIXELS:
        return None, n

    hh = stack["HH"].values[:, mask]
    hv = stack["HV"].values[:, mask]
    with np.errstate(all="ignore"):
        hh_med = np.nanmedian(np.where(hh > 0, hh, np.nan), axis=1)
        hv_med = np.nanmedian(np.where(hv > 0, hv, np.nan), axis=1)
    return (to_db(hh_med), to_db(hv_med)), n


def classify(hh, hv, early):
    """Простые правила по ряду. Возвращает (метка, подробности)."""
    info = {"pulses": [], "harvest_idx": None}
    if np.all(np.isnan(hh)):
        return "unclear", info

    if np.nanmax(hv) < NOT_RICE_HV_MAX:
        return "not_rice", info
    if np.nanmax(hv[~early]) - np.nanmin(hv[early]) < RICE_HV_GROWTH:
        return "not_rice", info

    # Уборка: после максимума HV — первое падение HV на HARVEST_HV_DROP
    peak = int(np.nanargmax(hv))
    harvest = None
    for i in range(peak + 1, len(hv)):
        if hv[peak] - hv[i] >= HARVEST_HV_DROP:
            harvest = i
            break
    info["harvest_idx"] = harvest
    end = harvest if harvest is not None else len(hh)

    # Импульсы: падение HH, затем рост (сброс воды → повторный залив) до уборки
    for i in range(1, end - 1):
        if np.isnan(hh[i - 1]) or np.isnan(hh[i]):
            continue
        # только когда рис уже вырос (после EARLY_END): до этого падение HH = залив, а не сброс
        if early[i - 1]:
            continue
        drop = hh[i - 1] - hh[i]
        # не одиночный всплеск: точка должна быть ниже и той, что две даты назад
        if i >= 2 and np.isfinite(hh[i - 2]) and hh[i - 2] - hh[i] < DROP_DB / 2:
            continue
        if drop >= DROP_DB:
            later = hh[i + 1:end]
            if later.size and np.nanmax(later) - hh[i] >= RISE_DB:
                info["pulses"].append(i)
    return ("pulse" if info["pulses"] else "flat"), info


def plot_field(fid, dates, hh, hv, label, info, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    name, color = LABELS[label]
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=120)
    ax.plot(dates, hh, "-o", color="#c2185b", label="HH — вода и почва")
    ax.plot(dates, hv, "-s", color="#2e7d32", label="HV — растения", alpha=0.8)
    for i in info["pulses"]:
        ax.axvline(dates[i], color="#f2a900", lw=2, alpha=0.6)
        ax.annotate("сброс воды", (dates[i], hh[i]), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8, color="#b07a00")
    if info["harvest_idx"] is not None:
        ax.axvspan(dates[info["harvest_idx"]], dates[-1], color="#999", alpha=0.15)
        ax.text(dates[info["harvest_idx"]], ax.get_ylim()[1], " уборка", va="top", fontsize=8)
    ax.set_title(f"Поле {fid}: {name}", fontsize=10, color=color)
    ax.set_ylabel("сигнал радара, дБ")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    add_credit(fig)
    fig.savefig(path)
    plt.close(fig)


def compare_plot(rows, dates, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    flat = [r for r in rows if r["label"] == "flat"]
    pulse = [r for r in rows if r["label"] == "pulse"]
    if not flat or not pulse:
        print("! для compare.png нужно хотя бы по одному полю «ровно» и «ритм»")
        return
    f = min(flat, key=lambda r: np.nanmax(-np.diff(r["hh"][3:])) - 0.3 * (np.nanmax(r["hh"]) - np.nanmin(r["hh"])))  # самое гладкое, с ярким ростом
    p = max(pulse, key=lambda r: len(r["info"]["pulses"]))
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=130)
    ax.plot(dates, f["hh"], "-o", color=LABELS["flat"][1], label=f"Поле {f['id']}: вода стояла весь сезон")
    ax.plot(dates, p["hh"], "-o", color=LABELS["pulse"][1], label=f"Поле {p['id']}: воду сбрасывали")
    ax.set_title("Ритм воды на двух рисовых полях (NISAR, канал HH)")
    ax.set_ylabel("сигнал радара HH, дБ")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    add_credit(fig)
    fig.savefig(path)
    plt.close(fig)
    print(f"Сравнение двух полей: {path}")


def make_sound(hh, path, step=0.35, sr=22050):
    """Ритм поля → звук: выше сигнал HH — выше тон."""
    import wave
    v = np.array(hh, dtype=float)
    if np.all(np.isnan(v)):
        return
    v = np.nan_to_num(v, nan=np.nanmin(v))
    norm = (v - v.min()) / (v.max() - v.min() + 1e-9)
    t = np.linspace(0, step, int(sr * step), endpoint=False)
    env = np.minimum(1, np.minimum(t / 0.02, (step - t) / 0.05))
    chunks = [0.4 * env * np.sin(2 * np.pi * (220 + 660 * n) * t) for n in norm]
    audio = (np.concatenate(chunks) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())


def make_map(rows, stack, out):
    import folium
    from shapely.geometry import mapping
    from pyproj import Transformer

    epsg = int(stack.attrs["epsg"])
    to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    x, y = stack.x.values, stack.y.values
    lons, lats = to_ll.transform([x.min(), x.max()], [y.min(), y.max()])
    center = [float(np.mean(lats)), float(np.mean(lons))]

    m = folium.Map(location=center, zoom_start=12, tiles=None)
    folium.TileLayer("OpenStreetMap", name="Карта").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery | " + CREDIT, name="Спутник (оптика)").add_to(m)

    # Подложка: радарный кадр середины лета
    ql_dir = out.parent / "data" / "quicklooks"
    times = [np.datetime64(t, "D") for t in stack.time.values]
    target = np.datetime64("2026-08-01")
    if ql_dir.exists() and times:
        k = int(np.argmin([abs((t - target).astype(int)) for t in times]))
        hh = to_db(stack["HH"].values[k])
        hv = to_db(stack["HV"].values[k])
        def norm(a, lo, hi):
            return np.clip((a - lo) / (hi - lo), 0, 1)
        rgb = np.nan_to_num(np.dstack([norm(hh, -25, 0), norm(hv, -32, -8), norm(hh, -25, 0)]))
        if y[0] < y[-1]:
            rgb = rgb[::-1]
        folium.raster_layers.ImageOverlay(
            image=rgb, bounds=[[min(lats), min(lons)], [max(lats), max(lons)]],
            opacity=0.75, name=f"Радар NISAR {str(times[k])}").add_to(m)

    from shapely.geometry import mapping as _m
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": _m(r["geom"]),
         "properties": {"id": r["id"], "name": LABELS[r["label"]][0], "color": LABELS[r["label"]][1]}}
        for r in rows if r.get("chart") is None and r["label"] != "not_rice"]}
    if fc["features"]:
        folium.GeoJson(fc, name="Все поля",
                       style_function=lambda f: {"color": f["properties"]["color"], "weight": 0.5,
                                                 "fillColor": f["properties"]["color"], "fillOpacity": 0.45},
                       tooltip=folium.GeoJsonTooltip(fields=["id", "name"], labels=False)).add_to(m)
    for r in rows:
        if r.get("chart") is None:
            continue
        name, color = LABELS[r["label"]]
        img = base64.b64encode(Path(r["chart"]).read_bytes()).decode()
        html = (f"<b>Поле {r['id']}</b><br>{name}<br>"
                f"сбросов воды: {len(r['info']['pulses'])}<br>"
                f"<img src='data:image/png;base64,{img}' width='420'>")
        folium.GeoJson(
            mapping(r["geom"]),
            style_function=lambda _, c=color: {"color": c, "weight": 2, "fillColor": c, "fillOpacity": 0.45},
            tooltip=f"{r['id']}: {name}",
            popup=folium.Popup(html, max_width=450),
        ).add_to(m)

    legend = "".join(
        f"<div><span style='background:{c};width:12px;height:12px;display:inline-block;margin-right:6px'></span>{n}</div>"
        for k, (n, c) in LABELS.items() if any(r["label"] == k for r in rows))
    m.get_root().html.add_child(folium.Element(
        "<div style='position:fixed;bottom:20px;left:20px;z-index:9999;background:white;"
        "padding:10px;border-radius:8px;font:13px sans-serif;box-shadow:0 1px 4px #0003'>"
        f"<b>Rice Rhythm · NISAR L-band</b>{legend}<div style='font-size:10px;color:#555;margin-top:4px'>{CREDIT}<br>клик по полю с обводкой — график</div></div>"))
    folium.LayerControl().add_to(m)
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    path = out / "map.html"
    m.save(str(path))
    print(f"Карта: {path}  (открыть в браузере)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="data/stack.nc")
    ap.add_argument("--fields", default="fields.geojson")
    ap.add_argument("--out", default="out")
    ap.add_argument("--sound", action="store_true")
    ap.add_argument("--early-end", default=None, help="конец «начала сезона», ММ-ДД (по умолчанию 07-15)")
    ap.add_argument("--dsr", action="store_true",
                    help="регион, где практикуют прямой посев риса (Харьяна): растущие поля без весенней воды = DSR; "
                         "без флага такие поля = «другие культуры»")
    args = ap.parse_args()

    import pandas as pd
    import xarray as xr

    stack = xr.open_dataset(args.stack, engine="h5netcdf").load()
    dates = pd.to_datetime(stack.time.values).to_pydatetime()
    out = Path(args.out)
    (out / "charts").mkdir(parents=True, exist_ok=True)

    rows, records = [], []
    fpath = Path(args.fields)
    if fpath.exists():
        series = []
        for fid, geom in load_fields(fpath, stack):
            res, n = field_series(stack, geom)
            if res is not None:
                series.append((fid, geom, res[0], res[1], n))
    else:
        print(f"! {fpath} не найден — режу весь участок на квадраты {GRID_CELL_M} м (план Б)")
        series = grid_series(stack)
    global DROP_DB, RISE_DB, EARLY_END
    if args.early_end:
        EARLY_END = tuple(int(v) for v in args.early_end.split("-"))
    early = np.array([(d.month, d.day) <= EARLY_END for d in dates])
    HH = np.array([s_[2] for s_ in series])
    HV = np.array([s_[3] for s_ in series])
    # общий для всего участка сдвиг по датам (погода, влажность почвы) — вычитаем
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        common = np.nanmedian(HH, axis=0)
        common = common - np.nanmean(common)
        growth = np.nanmax(HV[:, ~early], axis=1) - np.nanmin(HV[:, early], axis=1)
    HHc = HH - common
    noise = np.abs(np.diff(HHc[growth < RICE_HV_GROWTH], axis=1))
    noise = noise[np.isfinite(noise)]
    if DROP_DB is None:
        DROP_DB = round(float(np.percentile(noise, 99)), 1)
    if RISE_DB is None:
        RISE_DB = round(float(np.percentile(noise, 95)), 1)
    print(f"Шум |ΔHH| (нерис, после вычета общего сдвига): p95={np.percentile(noise, 95):.2f}, "
          f"p99={np.percentile(noise, 99):.2f} → DROP_DB={DROP_DB}, RISE_DB={RISE_DB}")
    for (fid, geom, hh, hv, n), hhc in zip(series, HHc):
        label, info = classify(hhc, hv, early)
        rows.append({"id": fid, "geom": geom, "hh": hh, "hv": hv, "label": label,
                     "info": info, "n_pix": n, "chart": None})

    # Прямой посев: рис есть (растёт HV), но в начале сезона нет открытой воды (HH не падает
    # ниже WATER_HH_DB), хотя у заметной части рисовых полей вокруг вода в это время есть.
    rice = [r for r in rows if r["label"] in ("flat", "pulse")]
    dry_frac = None
    if early.any() and rice:
        for r in rice:
            r["early_water"] = bool(np.nanmin(r["hh"][early]) < WATER_HH_DB)
        dry_frac = np.mean([r["early_water"] for r in rice])
        print(f"Рисовых полей с открытой водой в начале сезона (до {EARLY_END[1]:02d}.{EARLY_END[0]:02d}): {dry_frac:.0%}")
        for r in rice:
            if not r["early_water"]:
                r["label"] = "dry_seed" if (args.dsr and dry_frac >= DRY_SEED_MIN_FRAC) else "other_crop"
    stack.attrs["dry_frac"] = -1 if dry_frac is None else float(dry_frac)

    # Графики — не для всех тысяч квадратов, а по MAX_CHARTS_PER_LABEL на категорию
    rng = np.random.default_rng(0)
    for k in LABELS:
        grp = [r for r in rows if r["label"] == k]
        if k == "pulse":
            grp.sort(key=lambda r: -len(r["info"]["pulses"]))
            pick = grp[:MAX_CHARTS_PER_LABEL]
        else:
            pick = [grp[i] for i in rng.permutation(len(grp))[:MAX_CHARTS_PER_LABEL]]
        for r in pick:
            r["chart"] = out / "charts" / f"{r['id']}.png"
            plot_field(r["id"], dates, r["hh"], r["hv"], r["label"], r["info"], r["chart"])
    for r in rows:
        for d, a, b in zip(dates, r["hh"], r["hv"]):
            records.append({"field": r["id"], "date": d.date(), "HH_db": round(float(a), 2),
                            "HV_db": round(float(b), 2), "label": r["label"]})

    if not rows:
        raise SystemExit("Ни одного поля не обработано — проверьте fields.geojson и участок.")

    pd.DataFrame(records).to_csv(out / "timeseries.csv", index=False)

    feats = [{"type": "Feature", "geometry": r["geom"].__geo_interface__,
              "properties": {"id": r["id"], "label": r["label"], "label_ru": LABELS[r["label"]][0],
                             "pulses": len(r["info"]["pulses"]), "pixels": r["n_pix"]}} for r in rows]
    (out / "fields_result.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False))

    compare_plot(rows, dates, out / "compare.png")
    make_map(rows, stack, out)

    if args.sound:
        (out / "sound").mkdir(exist_ok=True)
        for r in rows:
            if r.get("chart") is not None:
                make_sound(r["hh"], out / "sound" / f"{r['id']}.wav")
        print(f"Звуки: {out / 'sound'}")

    counts = {k: sum(r["label"] == k for r in rows) for k in LABELS}
    lines = [
        "Rice Rhythm — итог",
        f"Снимков NISAR: {len(dates)} ({dates[0]:%d.%m.%Y} — {dates[-1]:%d.%m.%Y}), трек {stack.attrs.get('track', '?')}",
        f"Полей проанализировано: {len(rows)}",
    ] + [f"  {LABELS[k][0]}: {v} ({v / len(rows):.0%})" for k, v in counts.items()]
    rice_keys = ("flat", "pulse", "dry_seed")
    nr = sum(counts[k] for k in rice_keys)
    if nr:
        lines.append(f"Среди рисовых полей ({nr}): " + ", ".join(
            f"{LABELS[k][0].lower()} — {counts[k] / nr:.0%}" for k in rice_keys if counts[k] or k != "dry_seed"))
    lines.append(f"Пороги: DROP_DB={DROP_DB}, RISE_DB={RISE_DB}, NOT_RICE_HV_MAX={NOT_RICE_HV_MAX}, "
                 f"RICE_HV_GROWTH={RICE_HV_GROWTH}, WATER_HH_DB={WATER_HH_DB}")
    lines.append(CREDIT)
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
