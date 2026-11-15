#!/usr/bin/env python3
"""Ручная проверка: 5 случайных графиков каждой категории → out/check_<label>.png
  python montage.py out
"""
import json, sys
from pathlib import Path
import numpy as np
import imageio.v2 as imageio

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
g = json.loads((out / "fields_result.geojson").read_text())
by = {}
for f in g["features"]:
    p = out / "charts" / f"{f['properties']['id']}.png"
    if p.exists():
        by.setdefault(f["properties"]["label"], []).append(p)
rng = np.random.default_rng(1)
for lab, ps in by.items():
    pick = [ps[i] for i in rng.permutation(len(ps))[:5]]
    imgs = [imageio.imread(p)[..., :3] for p in pick]
    h, w = min(i.shape[0] for i in imgs), min(i.shape[1] for i in imgs)
    imgs = [i[:h, :w] for i in imgs]
    while len(imgs) < 6:
        imgs.append(np.full_like(imgs[0], 255))
    grid = np.vstack([np.hstack(imgs[0:3]), np.hstack(imgs[3:6])])
    imageio.imwrite(out / f"check_{lab}.png", grid)
    print(lab, len(ps), "→", out / f"check_{lab}.png")
