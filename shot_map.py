#!/usr/bin/env python3
"""PNG-скриншот out*/map.html (headless, через установленный Chromium-браузер)."""
import sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright
src = Path(sys.argv[1]).resolve(); dst = sys.argv[2]
exe = sys.argv[3] if len(sys.argv) > 3 else "/opt/brave.com/brave/brave"
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=exe, headless=True)
    pg = b.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto(src.as_uri(), wait_until="load", timeout=120000)
    time.sleep(8)  # дать подгрузиться спутниковым тайлам
    pg.screenshot(path=dst)
    b.close()
print("→", dst)
