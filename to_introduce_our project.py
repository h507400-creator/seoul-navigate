# make_map.py — 발표용 인터랙티브 지도
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium.plugins import Fullscreen
import os, time

BASE  = r"C:\전전심설\hw3\team_project"
EDGES = os.path.join(BASE, "seoul_edges_final.gpkg")
OUT   = os.path.join(BASE, "seoul_map.html")

SIMPLIFY = 8      # 선 단순화 (m)
MIN_LEN  = 100    # 이보다 짧은 구간 제외

# (컬럼, 표시명, 색, 임계값) — 임계값은 절대 기준
LAYERS = [
    ("fun_숲길",   "숲길 드라이브", "#1b7837", 0.70),
    ("culture_s", "문화·볼거리",   "#762a83", 0.55),
    ("tree_s",    "가로수",        "#7fbc41", 0.85),
    ("green_s",   "녹지",          "#276419", 0.85),
    ("water_s",   "수변",          "#2166ac", 0.85),
    ("curve_s",   "굴곡",          "#e08214", 0.90),
    ("grade_s",   "경사",          "#b2182b", 0.85),
    ("smooth_s",  "교통 흐름 원활",     "#4d4d4d", 0.80),
]

print("데이터 로드 중...")
t0 = time.time()
e = gpd.read_file(EDGES)
e = e[e["length"] > MIN_LEN]
print(f"  {len(e):,}개 ({time.time()-t0:.0f}초)")

e = e.to_crs(4326)
e["geometry"] = e.geometry.simplify(SIMPLIFY / 111000)

m = folium.Map(location=[37.5545, 126.9707], zoom_start=12,
               tiles="OpenStreetMap", prefer_canvas=True)
Fullscreen().add_to(m)

print("\n레이어 생성:")
for col, label, color, thr in LAYERS:
    if col not in e.columns:
        print(f"  {label}: 컬럼 없음 — 건너뜀")
        continue
    sub = e[e[col] >= thr].copy()
    pct = len(sub) / len(e) * 100
    if len(sub) == 0:
        print(f"  {label}: 해당 없음 (임계 {thr})")
        continue

    fg = folium.FeatureGroup(
        name=f"{label} ≥{thr} ({len(sub):,}개 · {pct:.1f}%)",
        show=(col == "fun_숲길"))

    lo, hi = sub[col].min(), sub[col].max()
    rng = max(hi - lo, 1e-9)

    for r in sub.itertuples():
        g = r.geometry
        if g.geom_type != "LineString":
            continue
        v = (getattr(r, col) - lo) / rng
        nm = getattr(r, "road_name", "") or "(이름없음)"
        folium.PolyLine(
            [(y, x) for x, y in g.coords],
            color=color,
            weight=2 + v * 3,
            opacity=0.5 + v * 0.45,
            tooltip=f"{nm} · {label} {getattr(r, col):.2f}",
        ).add_to(fg)
    fg.add_to(m)
    print(f"  {label}: {len(sub):,}개 ({pct:.1f}%)")

folium.LayerControl(collapsed=False).add_to(m)
m.save(OUT)
print(f"\n저장: {OUT}  ({os.path.getsize(OUT)/1e6:.1f}MB)")