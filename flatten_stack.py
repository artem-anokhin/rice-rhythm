#!/usr/bin/env python3
"""Убирает перепад яркости поперёк полосы съёмки (по колонкам), который в разные даты разный.

Для каждой даты и канала: медиана по колонкам (сглаженная) → вычитаем её отклонение от общей медианы.
Колонки, где HV на уровне шума (край полосы), в эту дату помечаются как нет данных.
  python flatten_stack.py data_karnal/stack.nc data_karnal/stack_flat.nc [--check]
"""
import sys
import warnings

import numpy as np
import xarray as xr
from scipy.ndimage import median_filter

warnings.filterwarnings("ignore")
NOISE_HV_DB = -27.0   # медиана HV по колонке ниже — край полосы/шум
WIN = 51              # сглаживание профиля, колонок


def db(a):
    return 10 * np.log10(np.where(a > 0, a, np.nan))


def profiles(st):
    out = {}
    for v in ("HH", "HV"):
        A = db(st[v].values)
        out[v] = np.nanmedian(A, axis=1)   # (time, x)
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    st = xr.open_dataset(src, engine="h5netcdf").load()
    pr = profiles(st)
    T = st.sizes["time"]
    for t in range(T):
        hv = pr["HV"][t]
        s = lambda a: np.nanpercentile(a, 90) - np.nanpercentile(a, 10)
        print(f"{str(st.time.values[t])[:10]}: разброс медианы HV по колонкам p90−p10 = {s(hv):.1f} дБ, "
              f"колонок на уровне шума: {np.mean(hv < NOISE_HV_DB):.0%}")
    if dst is None:
        return
    for v in ("HH", "HV"):
        A = db(st[v].values)
        for t in range(T):
            p = pr[v][t].copy()
            bad = pr["HV"][t] < NOISE_HV_DB
            p[bad] = np.nan
            ok = np.isfinite(p)
            if ok.sum() < 10:
                continue
            fill = np.interp(np.arange(len(p)), np.where(ok)[0], p[ok])
            sm = median_filter(fill, size=WIN, mode="nearest")
            A[t] = A[t] - (sm - np.nanmedian(p[ok]))[None, :]
            A[t][:, bad] = np.nan
        st[v].values[:] = (10 ** (A / 10)).astype("float32")
    st.attrs["flattened"] = "per-date across-track column profile removed; noise-floor columns masked"
    st.to_netcdf(dst, engine="h5netcdf")
    print("→", dst)
    pr2 = profiles(st)
    for t in range(T):
        print(f"  после: {str(st.time.values[t])[:10]} разброс HV по колонкам "
              f"{np.nanpercentile(pr2['HV'][t], 90) - np.nanpercentile(pr2['HV'][t], 10):.1f} дБ")


if __name__ == "__main__":
    main()
