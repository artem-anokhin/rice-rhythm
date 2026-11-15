#!/usr/bin/env python3
"""Sentinel-2 MNDWI на станциях Humnoke: доля пикселей воды в окне ~100 м (5×5 пикс. по 20 м)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pystac_client, planetary_computer, rasterio
from rasterio.warp import transform as wtr
ST = {"US-HRA": (34.5852, -91.7517), "US-HRC": (34.5888, -91.7517)}
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
items = sorted(cat.search(collections=["sentinel-2-l2a"], bbox=[-91.76, 34.58, -91.74, 34.595],
               datetime="2026-06-10/2026-10-31", query={"eo:cloud_cover": {"lt": 20}}).items(), key=lambda i: i.datetime)
print("сцен:", len(items))
seen = set()
for it in items:
    d = it.datetime.strftime("%d.%m")
    if d in seen: continue
    out = []
    try:
        for s, (lat, lon) in ST.items():
            v = {}
            for b in ("B03", "B11", "SCL"):
                with rasterio.open(it.assets[b].href) as src:
                    xs, ys = wtr("EPSG:4326", src.crs, [lon], [lat])
                    r, c = src.index(xs[0], ys[0])
                    h = 2 if src.res[0] > 15 else 5
                    a = src.read(1, window=((r - h, r + h + 1), (c - h, c + h + 1))).astype(float)
                    v[b] = a
            g = v["B03"]; g = g.reshape(5, 2, 5, 2).mean((1, 3)) if g.shape[0] == 11 and False else g[::2, ::2][:5, :5]
            sw = v["B11"][:5, :5]; scl = v["SCL"][:5, :5]
            clear = np.isin(scl, [4, 5, 6, 7])
            if clear.mean() < 0.8: out.append(f"{s}: облака"); continue
            m = (g - sw) / np.maximum(g + sw, 1)
            out.append(f"{s}: вода {np.mean(m[clear] > 0):.0%} (MNDWI {np.median(m[clear]):+.2f})")
    except Exception as e:
        out.append(f"ошибка {type(e).__name__}")
    seen.add(d)
    print(d, " | ".join(out), flush=True)
