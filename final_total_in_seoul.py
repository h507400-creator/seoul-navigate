from pyrosm import OSM
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import glob
import time
import os
from shapely.geometry import LineString
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE   = r"C:\전전심설\hw3\team_project"
PBF    = os.path.join(BASE, "south-korea-261104.osm.pbf")
DEM_F  = os.path.join(BASE, "seoul_dem.tif")
TREE_F = os.path.join(BASE, "2026 서울시 가로수 위치정보.xlsx")
SIG_F  = os.path.join(BASE, "서울시 교차로 및 횡단보도 시설 위치정보_20260824.xlsx")

SEOUL   = [126.76, 37.41, 127.19, 37.70]
CRS_M   = 5186
BUF     = 150      # 도로 버퍼 (m)
TREE_BUF = 30      # 가로수 집계 반경 (m)
SIG_BUF = 150      # 신호 교차로 집계 반경 (m)
WMAX    = 500      # 수변 영향 최대거리 (m)
WMIN_A  = 20000     # 이보다 작은 수변 폴리곤 제외 (m²)
STEP    = 20       # 굴곡도 리샘플 간격 (m)
CUL_MAX = 200      # 문화시설 반경(m)

DROP_HW = ["motorway", "motorway_link", "trunk", "trunk_link"]

HW_RANK = {
    "residential": 1, "living_street": 1, "unclassified": 1,
    "tertiary": 2, "tertiary_link": 2,
    "secondary": 3, "secondary_link": 3,
    "primary": 4, "primary_link": 4,
    "trunk": 5, "trunk_link": 5, "motorway": 5, "motorway_link": 5,
}

# ---------- 공통 함수 ----------
def to_score(s):
    """0보다 큰 값들끼리만 백분위 순위. 0점은 0으로 유지."""
    out = pd.Series(0.0, index=s.index)
    nz = s > 0
    if nz.any():
        out[nz] = s[nz].rank(pct=True)
    return out

def resample(line, step=STEP):
    n = max(int(line.length // step), 2)
    return LineString([line.interpolate(i / n, normalized=True)
                       for i in range(n + 1)])

def curvature(geom):
    """방향 변화각 누적 / 길이 (rad/m)"""
    if geom is None or geom.is_empty or geom.geom_type != "LineString":
        return 0.0
    if geom.length < 10:
        return 0.0
    xy = np.array(resample(geom).coords)
    if len(xy) < 3:
        return 0.0
    d = np.diff(xy, axis=0)
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return float(np.sum(np.abs(np.diff(ang))) / geom.length)

def count_in_buffer(points, buf, idx, crs):
    if len(points) == 0:
        return pd.Series(0.0, index=idx)
    bg = gpd.GeoDataFrame({"eid": range(len(idx))}, geometry=buf.values, crs=crs)
    hit = gpd.sjoin(points[["geometry"]], bg, predicate="within")
    cnt = hit.groupby("eid").size()
    return pd.Series(cnt.reindex(range(len(idx)), fill_value=0).values, index=idx)

def dem_elevation(lons, lats):
    """로컬 DEM에서 고도 읽기 (WGS84 경위도)"""
    with rasterio.open(DEM_F) as src:
        vals = list(src.sample([(lo, la) for lo, la in zip(lons, lats)]))
    e = np.array([v[0] for v in vals], dtype=float)
    e[(e < 0) | (e > 2000)] = np.nan
    return e

# ---------- 1. 도로망 ----------
print("=== 도로망 ===")
t0 = time.time()
osm = OSM(PBF, bounding_box=SEOUL)
edges = osm.get_network(network_type="driving")
print(f"원본 {len(edges):,}개 ({time.time()-t0:.0f}초)")

if "tunnel" in edges.columns:
    edges = edges[edges["tunnel"].isna()]

# 자동차전용도로는 데이터에 포함하되 모델 학습에서만 제외
edges["is_expwy"] = edges["highway"].isin(DROP_HW)
print(f"  자동차전용 {edges['is_expwy'].sum():,}개 (학습 제외, 경로 탐색에는 사용)")
edges = edges[edges.geom_type == "LineString"].reset_index(drop=True)
edges = edges.to_crs(CRS_M)
edges["length"] = edges.geometry.length
edges = edges[edges["length"] > 10].reset_index(drop=True)
print(f"정리 후 {len(edges):,}개 (평균 {edges['length'].mean():.0f}m)")

# ---------- 2. 녹지 · 수변 ----------
print("\n=== 녹지 · 수변 ===")
green = osm.get_data_by_custom_criteria(
    custom_filter={"leisure": ["park", "garden"],
                   "landuse": ["forest", "grass"],
                   "natural": ["wood"]},
    filter_type="keep", keep_nodes=False, keep_ways=True, keep_relations=True
).to_crs(CRS_M)

water = osm.get_data_by_custom_criteria(
    custom_filter={"natural": ["water"], "waterway": ["river"]},
    filter_type="keep", keep_nodes=False, keep_ways=True, keep_relations=True
).to_crs(CRS_M)

green = green[green.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
bad = ~green.geometry.is_valid
green.loc[bad, "geometry"] = green.loc[bad, "geometry"].buffer(0)
green = green[green.geometry.is_valid & ~green.geometry.is_empty]
green["geometry"] = green.geometry.simplify(10)
green = green[~green.geometry.is_empty].reset_index(drop=True)

# 면형만 사용 (선형 하천은 복개천이 섞여 구분 불가)
water = water[water.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
badw = ~water.geometry.is_valid
water.loc[badw, "geometry"] = water.loc[badw, "geometry"].buffer(0)
water = water[water.geometry.is_valid & ~water.geometry.is_empty]

# 인공 수조 제외
BAD_WATER = ["wastewater", "basin", "pool", "watertank", "fountain", "now empty"]
if "water" in water.columns:
    water = water[~water["water"].isin(BAD_WATER)]

# 면적 기준별 개수 확인
for a in [5000, 10000, 20000, 50000]:
    print(f"  {a:,}m² 이상: {(water.geometry.area >= a).sum()}개")

water = water[water.geometry.area >= WMIN_A].reset_index(drop=True)
if "name" in water.columns:
    print("  남은 수변:", water["name"].dropna().unique()[:25])
print(f"  수변 {len(water)}개")
print(water.columns.tolist())
print(water.geometry.area.sort_values(ascending=False).head(10) / 1e6)

# ---------- 3. 가로수 · 신호 교차로 ----------
print("\n=== 서울시 데이터 ===")
TREE_ALL = os.path.join(BASE, "trees_all.gpkg")
if os.path.exists(TREE_ALL):
    trees = gpd.read_file(TREE_ALL)
else:
    xl = pd.ExcelFile(TREE_F)
    dfs = []
    for sh in xl.sheet_names:
        try:
            dfs.append(pd.read_excel(TREE_F, sheet_name=sh, header=2))
        except Exception:
            pass
    t = pd.concat(dfs, ignore_index=True).dropna(subset=["좌표(경도)", "좌표(위도)"])
    trees = gpd.GeoDataFrame(
        t[["자치구", "노선", "수종"]],
        geometry=gpd.points_from_xy(t["좌표(경도)"], t["좌표(위도)"]),
        crs=4326).to_crs(CRS_M)
    trees.to_file(TREE_ALL, driver="GPKG")
print(f"가로수 {len(trees):,}그루")

SIG_ALL = os.path.join(BASE, "signals_all.gpkg")
if os.path.exists(SIG_ALL):
    signals = gpd.read_file(SIG_ALL)
else:
    s = pd.read_excel(SIG_F)
    s = s[s["보행등유무"] == "유"]
    s = s.groupby("교차로관리번호")[["X좌표", "Y좌표"]].mean().reset_index()
    signals = gpd.GeoDataFrame(
        s, geometry=gpd.points_from_xy(s["X좌표"], s["Y좌표"]), crs=CRS_M)
    signals.to_file(SIG_ALL, driver="GPKG")
print(f"신호 교차로 {len(signals):,}개")

# ---------- 4. 경사 ----------
print("\n=== 경사 ===")
t0 = time.time()
ends = edges.geometry.apply(lambda g: (g.coords[0], g.coords[-1]))
p0 = gpd.GeoSeries(gpd.points_from_xy([c[0][0] for c in ends],
                                      [c[0][1] for c in ends]), crs=CRS_M).to_crs(4326)
p1 = gpd.GeoSeries(gpd.points_from_xy([c[1][0] for c in ends],
                                      [c[1][1] for c in ends]), crs=CRS_M).to_crs(4326)
e0 = dem_elevation(p0.x.values, p0.y.values)
e1 = dem_elevation(p1.x.values, p1.y.values)
diff = np.abs(e0 - e1)
edges["grade"] = np.where(np.isnan(diff), 0.0, diff / edges["length"].values)
edges.loc[edges["length"] < 30, "grade"] = 0.0
edges["grade"] = edges["grade"].clip(0, 0.20)
allz = np.concatenate([e0, e1])
print(f"고도 {np.nanmin(allz):.0f}~{np.nanmax(allz):.0f}m "
      f"(결측 {np.isnan(allz).mean()*100:.1f}%)")
print(f"평균 경사 {edges['grade'].mean()*100:.2f}% ({time.time()-t0:.0f}초)")

# ---------- 5. 속성 계산 ----------
print("\n=== 속성 계산 ===")
t0 = time.time()
buf  = edges.geometry.buffer(BUF)
tbuf = edges.geometry.buffer(TREE_BUF)
sbuf = edges.geometry.buffer(SIG_BUF)
print(f"  버퍼 완료 ({time.time()-t0:.0f}초)")

# 녹지 — 겹치는 폴리곤만 교차
t1 = time.time()
buf_gdf = gpd.GeoDataFrame({"eid": range(len(edges))},
                           geometry=buf.values, crs=edges.crs)
hit = gpd.sjoin(buf_gdf, green[["geometry"]], predicate="intersects")
area = np.zeros(len(edges))
for eid, grp in hit.groupby("eid"):
    g = buf.iloc[eid]
    area[eid] = sum(g.intersection(green.geometry.iloc[i]).area
                    for i in grp["index_right"])
edges["green"] = np.clip(area / buf.area.values, 0, 1)
print(f"  녹지 완료 ({time.time()-t1:.0f}초)")

# 수변
t1 = time.time()
nr = gpd.sjoin_nearest(edges[["geometry"]], water[["geometry"]],
                       max_distance=WMAX, distance_col="dw")
dw = nr.groupby(nr.index)["dw"].min()
edges["water"] = ((1 - dw / WMAX).reindex(edges.index).fillna(0).clip(0, 1)) ** 2
print(f"  수변 완료 ({time.time()-t1:.0f}초)")

# 가로수 — 있어야 할 수 대비 실제 수 (8m 간격 × 양쪽)
t1 = time.time()
n_tree = count_in_buffer(trees, tbuf, edges.index, edges.crs)
expected = (edges["length"] / 8 * 2).clip(lower=1)
edges["tree"] = (n_tree / expected).clip(0, 2)
edges["signal"] = count_in_buffer(signals, sbuf, edges.index, edges.crs) / edges["length"] * 1000
print(f"  가로수·신호 완료 ({time.time()-t1:.0f}초)")


# 문화·역사 시설 밀도
t1 = time.time()
cul_pts = []
for tags in [{"historic": ["castle", "city_gate", "citywalls",
                           "monument", "archaeological_site", "ruins",
                           "palace", "temple", "shrine", "memorial", "yes"]},
             {"tourism": ["museum", "gallery", "attraction", "viewpoint"]}]:
    d = osm.get_data_by_custom_criteria(
        custom_filter=tags, filter_type="keep",
        keep_nodes=True, keep_ways=True, keep_relations=True)
    if d is not None and len(d):
        d = d.to_crs(CRS_M)
        d = d[~d.geometry.is_empty]
        d["geometry"] = d.geometry.centroid
        cul_pts.append(d[["geometry"]])

culture = gpd.GeoDataFrame(pd.concat(cul_pts, ignore_index=True),
                           geometry="geometry", crs=CRS_M)
near_c = gpd.sjoin_nearest(edges[["geometry"]], culture[["geometry"]],
                           max_distance=CUL_MAX, distance_col="dc")
dc = near_c.groupby(near_c.index)["dc"].min()
edges["culture"] = ((1 - dc / CUL_MAX)
                    .reindex(edges.index).fillna(0).clip(0, 1)) ** 2
print(f"  문화시설 {len(culture):,}개 / 0점 "
      f"{(edges['culture']==0).mean()*100:.1f}% ({time.time()-t1:.0f}초)")


# 굴곡
t1 = time.time()
edges["curve"] = [curvature(g) for g in edges.geometry.values]
print(f"  굴곡 완료 ({time.time()-t1:.0f}초)")

edges["road_rank"] = edges["highway"].map(HW_RANK).fillna(1).astype(int)
edges["road_name"] = edges["name"].fillna("")

# ---------- 5-2. 교통 속도 (서울시 차량통행속도) ----------
SPD_CACHE = os.path.join(BASE, "speed_byroad.csv")
if os.path.exists(SPD_CACHE):
    byroad = pd.read_csv(SPD_CACHE, index_col=0)
else:
    files = sorted(glob.glob(os.path.join(BASE, "*차량통행속도*.xlsx")))
    print(f"\n=== 교통 속도 ({len(files)}개 파일) ===")
    parts = []
    for f in files:
        d = pd.read_excel(f)
        parts.append(d)
        print(f"  {os.path.basename(f)}: {len(d):,}행")
    spd = pd.concat(parts, ignore_index=True)
    print(f"  합계 {len(spd):,}행 / 도로 {spd['도로명'].nunique()}개")

    HCOL = [c for c in spd.columns if str(c).endswith("시")]
    DAY  = [c for c in [f"~{h:02d}시" for h in range(10, 19)] if c in HCOL]

    is_wk = spd["요일"].isin(["토", "일"])
    wk  = spd[is_wk].groupby("도로명")[DAY].mean().mean(axis=1)
    var = spd[~is_wk].groupby("도로명")[HCOL].mean().std(axis=1)
    jam = (spd[HCOL] < 15).mean(axis=1).groupby(spd["도로명"]).mean()

    byroad = pd.DataFrame({"spd_wk": wk, "spd_var": var, "jam": jam})
    byroad.to_csv(SPD_CACHE)

print(f"\n속도 데이터 도로 {len(byroad)}개")
print(byroad.describe().round(2).to_string())

for c in ["spd_wk", "spd_var", "jam"]:
    edges[c] = edges["road_name"].map(byroad[c])

edges["has_spd"] = edges["spd_wk"].notna()
m = edges["has_spd"]
print(f"\n매칭 구간 {m.sum():,}/{len(edges):,} ({m.mean()*100:.1f}%) / "
      f"도로 {edges.loc[m, 'road_name'].nunique()}개")

# 결측은 중앙값으로 채움 (학습에서는 has_spd로 걸러냄)
for c in ["spd_wk", "spd_var", "jam"]:
    edges[c] = edges[c].fillna(byroad[c].median())

print(f"\n속성 계산 전체 {(time.time()-t0)/60:.1f}분")


# ---------- 6. 정규화 ----------
for c in ["green", "water", "tree", "curve", "grade", "culture"]:
    edges[c + "_s"] = to_score(edges[c])

edges["signal_s"]   = edges["signal"].rank(pct=True)
edges["straight_s"] = np.where(edges["curve"] > 0, 1 - edges["curve_s"], 1.0)
edges["flow_s"]     = 1 - edges["signal_s"]

# 속도 관련: 데이터 있는 구간끼리만 백분위, 없는 곳은 0.5
for src, dst, invert in [("jam", "smooth_s", True),
                         ("spd_var", "stable_s", True),
                         ("spd_wk", "spd_wk_s", False)]:
    v = pd.Series(0.5, index=edges.index)
    sub = edges.loc[edges["has_spd"], src].rank(pct=True)
    v[edges["has_spd"]] = (1 - sub) if invert else sub
    edges[dst] = v

print("\n[0점 비율]")
for c, l in [("green", "녹지"), ("water", "수변"), ("tree", "가로수"),
             ("signal", "신호"), ("curve", "굴곡"), ("grade", "경사")]:
    print(f"  {l}: {(edges[c] == 0).mean()*100:.1f}%")

edges.to_file(os.path.join(BASE, "seoul_edges_raw.gpkg"), driver="GPKG")
print("\n원시 저장: seoul_edges_raw.gpkg")


# ---------- 가로수 확인 ----------
print(f"\n[가로수 점검]")
print(f"  총 {len(trees):,}그루 / 자치구 {trees['자치구'].nunique()}개")
print(trees["자치구"].value_counts().head(8).to_string())
print(f"  0점 비율 {(edges['tree']==0).mean()*100:.1f}%")

print("\n  [도로별 검증]")
for rn in ["압구정로", "퇴계로", "세종대로", "테헤란로", "올림픽로"]:
    sub = edges[edges["road_name"] == rn]
    if len(sub):
        print(f"    {rn:8s} 구간 {len(sub):3d}개 / "
              f"tree 평균 {sub['tree'].mean():.3f} / "
              f"0점 {(sub['tree']==0).mean()*100:3.0f}%")
# ---------- 7. 라벨 ----------
ROAD_TYPES = {
    "숲길": ["북악산로", "인왕산로", "창의문로", "자하문로",
            "소월로", "소파로", "워커힐로", "낙산길"],
    "도심": [       
         # 사대문 (문화·역사)
        "북촌로", "삼청로", "정동길", "덕수궁길", "율곡로", "사직로", "대학로",
        # 용산·마포 (분위기·상권)
        "이태원로", "양화로", "효창원로",
        # 강남 (가로수·개방감)
        "압구정로", "도산대로", "선릉로", "양재천로",
        # 동남 (공원·호수)
        "올림픽로", "석촌호수로",
        # 서남 (수변·업무)
        "여의대로", "안양천로",
        # 수변
        "청계천로",],
}

FEAT_SET = {
    "숲길": ["curve_s", "grade_s", "green_s", "smooth_s"],
    "도심": ["tree_s", "green_s", "culture_s"],
}

name2type = {r: t for t, rs in ROAD_TYPES.items() for r in rs}
edges["rtype"] = edges["road_name"].map(name2type).fillna("")

print(f"\n라벨 {(edges['rtype'] != '').sum()}개")
print(edges[edges["rtype"] != ""].groupby("rtype").size())

print("\n[유형별 도로 — 데이터 보유 현황]")
have = set(edges["road_name"])
for t, rs in ROAD_TYPES.items():
    got = [r for r in rs if r in have]
    no  = [r for r in rs if r not in have]
    print(f"  {t}: {len(got)}/{len(rs)}개")
    if no:
        print(f"    없음: {no}")

print("\n[도로별 구간 수]")
print(edges.loc[edges["rtype"] != "", ["rtype", "road_name"]]
      .value_counts().to_string())

print("\n[문화시설 검증]")
for rn in ["북촌로", "삼청로", "덕수궁길", "대학로", "이태원로",
           "테헤란로", "올림픽로"]:
    sub = edges[edges["road_name"] == rn]
    if len(sub):
        print(f"  {rn:8s} culture {sub['culture'].mean():6.2f} / "
              f"정규화 {sub['culture_s'].mean():.3f}")

print("\n[유형별 평균 점수]")
allf = ["curve_s", "grade_s", "green_s", "tree_s", "water_s",
        "flow_s", "smooth_s", "stable_s"]
sub = edges[edges["has_spd"]]
cmp_ = sub[sub["rtype"] != ""].groupby("rtype")[allf].mean()
cmp_.loc["(비라벨)"] = sub[sub["rtype"] == ""][allf].mean()
print(cmp_.round(3).T)

# ---------- 8. 유형별 모델 ----------
def train_type(rtype, feats, seed=42, verbose=True):
    base = edges[(edges["length"] > 50) & (~edges["is_expwy"])]
    pos  = base[base["rtype"] == rtype]

    # 속도 변수를 쓰면 데이터 있는 도로끼리만 비교 (데이터 유무 교란 차단)
    use_spd = any(f in feats for f in ["smooth_s", "stable_s", "spd_wk_s"])
    cand = base[base["rtype"] == ""]
    if use_spd:
        cand = cand[cand["has_spd"]]

    if pos["road_name"].nunique() < 6 or len(pos) < 30 or len(cand) < 100:
        if verbose:
            print(f"\n[{rtype}] 라벨/음성 부족 "
                  f"(양성 {len(pos)} / 음성후보 {len(cand)}) — 건너뜀")
        return None

    neg = pd.concat([
        cand[cand["road_rank"] == r].sample(
            min(n * 3, (cand["road_rank"] == r).sum()), random_state=seed)
        for r, n in pos["road_rank"].value_counts().items()
        if (cand["road_rank"] == r).sum() > 0
    ])
    d = pd.concat([pos, neg]).copy()
    d["lab"] = (d["rtype"] == rtype).astype(int)
    d["grp"] = np.where(d["road_name"] != "", d["road_name"],
                        "unnamed_" + d.index.astype(str))

    X, y, g = d[feats].values, d["lab"].values, d["grp"].values
    p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    nfold = min(5, pos["road_name"].nunique())

    if not verbose:
        try:
            return cross_val_score(p, X, y, cv=GroupKFold(nfold), groups=g,
                                   scoring="roc_auc").mean()
        except Exception:
            return None

    print(f"\n{'='*50}")
    print(f"[{rtype} 모델]  변수: {', '.join(feats)}")
    print(f"  음성 후보 {len(cand):,}개 (속도데이터 {'있는 것만' if use_spd else '전체'})")
    print(f"  학습 {len(d)}개 (양성 {y.sum()} / 음성 {len(y)-y.sum()}) "
          f"/ 도로 {pos['road_name'].nunique()}개 / {nfold}-fold")

    a = cross_val_score(p, X, y, cv=StratifiedKFold(5, shuffle=True,
                        random_state=42), scoring="roc_auc")
    b = cross_val_score(p, X, y, cv=GroupKFold(nfold), groups=g, scoring="roc_auc")
    m_ = cross_val_score(
        make_pipeline(StandardScaler(),
                      MLPClassifier(hidden_layer_sizes=(16,),
                                    max_iter=3000, random_state=42)),
        X, y, cv=GroupKFold(nfold), groups=g, scoring="roc_auc")
    print(f"  AUC  무작위 {a.mean():.3f} / 도로별 {b.mean():.3f} "
          f"(±{b.std():.3f}) / MLP {m_.mean():.3f}")

    p.fit(X, y)
    print("  계수:", "  ".join(
        f"{k} {v:+.2f}" for k, v in
        pd.Series(p.named_steps["logisticregression"].coef_[0],
                  index=feats).sort_values(key=abs, ascending=False).items()))

    print("  변수 제외 시 AUC:")
    for drop in feats:
        f2 = [c for c in feats if c != drop]
        if not f2:
            print(f"    {drop:12s}-단일변수라 제외 불가")
            continue
        s = cross_val_score(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            d[f2].values, y, cv=GroupKFold(nfold), groups=g, scoring="roc_auc")
        print(f"    {drop:12s} 제외 → {s.mean():.3f} ({s.mean()-b.mean():+.3f})")

    seeds = [train_type(rtype, feats, seed=s, verbose=False) for s in range(10)]
    seeds = [v for v in seeds if v is not None and not np.isnan(v)]
    if seeds:
        print(f"  시드 {len(seeds)}회 평균 {np.mean(seeds):.3f} "
              f"(±{np.std(seeds):.3f}, {min(seeds):.3f}~{max(seeds):.3f})")
    return p

pipes = {}
for rt, fs in FEAT_SET.items():
    p = train_type(rt, fs)
    if p is not None:
        pipes[rt] = p
FEAT_SET = {k: v for k, v in FEAT_SET.items() if k in pipes}
print(f"\n학습 완료: {list(pipes)}")

# ---------- 9. 점수 ----------
for rt, fs in FEAT_SET.items():
    edges[f"fun_{rt}"] = pipes[rt].predict_proba(edges[fs].values)[:, 1]

fcols = [f"fun_{rt}" for rt in FEAT_SET]
edges["fun"] = edges[fcols].max(axis=1)
edges["fun_type"] = edges[fcols].idxmax(axis=1).str[4:]

lg = edges[(edges["road_name"] != "") & (edges["length"] > 300)]
for rt in FEAT_SET:
    print(f"\n[{rt} 상위 15 (300m 이상)]")
    print(lg.nlargest(15, f"fun_{rt}")[["road_name", "rtype", "length",
                                        "jam", f"fun_{rt}"]].round(3).to_string())

print("\n[도로명별 점수 (총연장 1km 이상, 상위 25)]")
byname = edges[edges["road_name"] != ""].groupby("road_name").agg(
    **{rt: (f"fun_{rt}", "mean") for rt in FEAT_SET},
    jam=("jam", "mean"),
    km=("length", lambda s: s.sum() / 1000),
    rtype=("rtype", "first"))
byname["best"] = byname[list(FEAT_SET)].max(axis=1)
print(byname[byname["km"] > 1].nlargest(25, "best").round(3).to_string())

# ---------- 10. 노드 연결 정보 붙이기 ----------
print("\n=== 노드 연결 정보 ===")
t0 = time.time()
nodes_g, edges_g = osm.get_network(network_type="driving", nodes=True)
print(f"  노드 {len(nodes_g):,} / 분할 엣지 {len(edges_g):,} ({time.time()-t0:.0f}초)")

# OSM way id 기준으로 시작·끝 노드 추출
uv = edges_g.groupby("id").agg(u=("u", "first"), v=("v", "last"))
edges = edges.merge(uv, left_on="id", right_index=True, how="left")
ok = edges["u"].notna()
print(f"  u/v 매칭 {ok.sum():,}/{len(edges):,} ({ok.mean()*100:.1f}%)")

# 노드 좌표도 따로 저장 (그래프 복원용)
nodes_out = nodes_g[["id", "lon", "lat", "geometry"]].copy()
nodes_out.to_file(os.path.join(BASE, "seoul_nodes.gpkg"), driver="GPKG")
print(f"  노드 저장: seoul_nodes.gpkg ({len(nodes_out):,}개)")

# ---------- 10-2. 분할 엣지에 점수 붙이기 (경로 탐색용) ----------
print("\n=== 경로 탐색용 그래프 ===")
score_cols = ["road_name", "road_rank", "is_expwy",
              "curve_s", "straight_s", "grade_s",
              "green_s", "tree_s", "water_s", "culture_s",
              "flow_s", "smooth_s", "stable_s", "spd_wk", "jam",
              "fun", "fun_type"] + fcols
score_cols = [c for c in score_cols if c in edges.columns]

seg = edges_g.merge(edges[["id"] + score_cols], on="id", how="inner")
seg = seg.to_crs(CRS_M)
seg["length"] = seg.geometry.length
seg = seg[seg["length"] > 0].reset_index(drop=True)
print(f"  분할 엣지 {len(seg):,}개 (점수 있는 것만)")

# ---------- 11. 인계 저장 ----------
cols = [c for c in [
    "id", "u", "v", "road_name", "highway", "length", "road_rank", "oneway",
    "is_expwy", "curve_s", "straight_s", "grade_s", "green_s", "tree_s", "water_s",
    "culture_s", "flow_s", "smooth_s", "stable_s", "spd_wk_s",
    "spd_wk", "jam", "has_spd", "fun", "fun_type",
] + fcols + ["geometry"] if c in edges.columns]

out = edges[cols]
out.to_file(os.path.join(BASE, "seoul_edges_final.gpkg"), driver="GPKG")
print(f"\n인계 파일: seoul_edges_final.gpkg ({len(out):,}행 × {len(cols)}열)")
print("컬럼:", cols)

# 경로 탐색용 분할 파일
seg_cols = [c for c in ["id", "u", "v", "oneway", "highway", "length"]
            + score_cols + ["geometry"] if c in seg.columns]
seg[seg_cols].to_file(os.path.join(BASE, "seoul_segments_final.gpkg"),
                      driver="GPKG")
print(f"경로 단위 저장: seoul_segments_final.gpkg "
      f"({len(seg):,}행 × {len(seg_cols)}열)")
# ---------- 12. 그래프 복원 테스트 ----------
import networkx as nx
import random

print("\n=== 그래프 복원 테스트 ===")
G = nx.DiGraph()
n_one = n_rev = 0
for r in seg.itertuples():
    ow = getattr(r, "oneway", None)
    if isinstance(ow, bool):
        one, rev = ow, False
    else:
        s_ = str(ow).strip().lower()
        rev = (s_ == "-1")
        one = s_ in ("yes", "true", "1") or rev
    n_one += one
    n_rev += rev

    attrs = dict(length=r.length, fun=r.fun, tree=r.tree_s,
                 green=r.green_s, smooth=r.smooth_s, water=r.water_s,
                 curve=r.curve_s, grade=r.grade_s,
                 name=r.road_name if isinstance(r.road_name, str) else "")

    if rev:
        G.add_edge(int(r.v), int(r.u), **attrs)
    else:
        G.add_edge(int(r.u), int(r.v), **attrs)
        if not one:
            G.add_edge(int(r.v), int(r.u), **attrs)

print(f"  노드 {G.number_of_nodes():,} / 엣지 {G.number_of_edges():,}")
print(f"  일방통행 {n_one:,} ({n_one/len(seg)*100:.1f}%) / 역방향 {n_rev}")

und = G.to_undirected()
wcc = max(nx.connected_components(und), key=len)
print(f"  약한 연결 {len(wcc):,}개 ({len(wcc)/G.number_of_nodes()*100:.1f}%)")

scc = max(nx.strongly_connected_components(G), key=len)
print(f"  강한 연결 {len(scc):,}개 ({len(scc)/G.number_of_nodes()*100:.1f}%)")

big = scc        # 경로 탐색은 강한 연결 요소 안에서만

random.seed(0)
ok = 0
for i in range(5):
    a, b = random.sample(list(big), 2)
    try:
        p = nx.shortest_path(G, a, b, weight="length")
        d = sum(G[x][y]["length"] for x, y in zip(p, p[1:]))
        print(f"  테스트 {i+1}: OK — 노드 {len(p)}개, {d/1000:.1f}km")
        ok += 1
    except Exception as e:
        print(f"  테스트 {i+1}: 실패 — {e}")
print(f"  성공 {ok}/5")

# 비용 계산용 배열 (한 번만)
EDGE_LIST = list(G.edges())
LEN  = np.array([G.edges[e]["length"] for e in EDGE_LIST])
FEAT = {k: np.array([G.edges[e].get(k, 0.5) for e in EDGE_LIST])
        for k in ["fun", "tree", "green", "water", "smooth", "curve", "grade"]}
print(f"  비용 배열 준비 완료 ({len(LEN):,}개)")

# ---------- 13. 경로 탐색 검증 ----------
if ok > 0:
    MAX_DETOUR = 0.30        # 우회 허용 한도 30%

    def route(G, a, b, pref, base_len=None):
        pen = np.zeros(len(LEN))
        for key, w in pref.items():
            pen += w * (1 - FEAT[key])
        cost = LEN * (1 + pen)
        for e, c in zip(EDGE_LIST, cost):
            G.edges[e]["cost"] = c
        p = nx.shortest_path(G, a, b, weight="cost")
        pair = list(zip(p, p[1:]))
        d = sum(G[x][y]["length"] for x, y in pair)
        out = {"km": d / 1000,
               "fun":    np.mean([G[x][y]["fun"]    for x, y in pair]),
               "tree":   np.mean([G[x][y]["tree"]   for x, y in pair]),
               "green":  np.mean([G[x][y]["green"]  for x, y in pair]),
               "water":  np.mean([G[x][y]["water"]  for x, y in pair]),
               "smooth": np.mean([G[x][y]["smooth"] for x, y in pair])}
        out["detour"] = (d / base_len - 1) if base_len else 0.0
        return p, out

    def route_capped(G, a, b, pref, max_detour=MAX_DETOUR, iters=8):
        """우회율 상한을 지키며 선호를 최대한 반영 (가중치 이분탐색)"""
        base = nx.shortest_path_length(G, a, b, weight="length")
        if not pref:
            p, r = route(G, a, b, {}, base)
            return p, r, 0.0

        lo, hi = 0.0, 50.0
        best, best_r, best_k = None, None, 0.0
        for _ in range(iters):
            k = (lo + hi) / 2
            scaled = {key: w * k for key, w in pref.items()}
            p, r = route(G, a, b, scaled, base)
            if r["detour"] <= max_detour:
                best, best_r, best_k, lo = p, r, k, k
            else:
                hi = k
        if best is None:
            best, best_r = route(G, a, b, {}, base)
            best_k = 0.0
        return best, best_r, best_k

    PRESETS = {
        "최단":        {},
        "숲길 드라이브":  {"fun": 1},
        "가로수길":      {"tree": 1},
        "공원길":       {"green": 1},
        "강변길":       {"water": 1},
        "안 막히는 길":   {"smooth": 1},
    }

    # ---- (1) 가중치 민감도 ----
    print("\n=== 가중치 민감도 (임의 구간) ===")
    random.seed(1)
    a, b = random.sample(list(big), 2)
    base_len = nx.shortest_path_length(G, a, b, weight="length")
    print(f"  최단거리 {base_len/1000:.1f}km")
    for w in [0, 1, 3, 10, 30]:
        _, r = route(G, a, b, {"fun": w}, base_len)
        print(f"  w={w:3d}  {r['km']:5.1f}km (우회 {r['detour']*100:+5.0f}%) "
              f"fun {r['fun']:.3f}")

    # ---- (2) 지역별 프리셋 비교 ----
    print(f"\n=== 지역별 프리셋 비교 (우회 한도 {MAX_DETOUR*100:.0f}%) ===")
    REGIONS = {
        "북악·평창":   (37.58, 37.62, 126.95, 127.01),
        "강남":       (37.48, 37.53, 127.02, 127.08),
        "여의도·마포":  (37.52, 37.56, 126.88, 126.95),
        "광진·성동":   (37.53, 37.56, 127.05, 127.11),
        "강서·양천":   (37.53, 37.57, 126.81, 126.88),
        "도봉·노원":   (37.63, 37.68, 127.02, 127.08),
        "구로·금천":   (37.45, 37.50, 126.87, 126.93),
        "송파·강동":   (37.50, 37.55, 127.11, 127.17),
    }
    nd = nodes_g.set_index("id")

    for rname, (la0, la1, lo0, lo1) in REGIONS.items():
        ar = nd[(nd["lat"].between(la0, la1)) & (nd["lon"].between(lo0, lo1))]
        cn = [n for n in ar.index if n in big]
        if len(cn) < 2:
            print(f"\n[{rname}] 노드 부족 — 건너뜀")
            continue

        # 3~6km 떨어진 구간 찾기
        random.seed(7)
        a3 = b3 = None
        for _ in range(60):
            x, y = random.sample(cn, 2)
            d = nx.shortest_path_length(G, x, y, weight="length")
            if 3000 <= d <= 6000:
                a3, b3, base3 = x, y, d
                break
        if a3 is None:
            a3, b3 = random.sample(cn, 2)
            base3 = nx.shortest_path_length(G, a3, b3, weight="length")

        print(f"\n[{rname}] 노드 {len(cn):,}개 / 최단 {base3/1000:.1f}km")
        print(f"  {'프리셋':13s} {'거리':>7s} {'우회':>6s} {'fun':>6s} "
              f"{'tree':>6s} {'green':>6s} {'water':>6s} {'smooth':>7s} "
              f"{'겹침':>6s} {'w':>5s}")

        base_p, _ = route(G, a3, b3, {}, base3)
        base_set = set(zip(base_p, base_p[1:]))

        for nm, pref in PRESETS.items():
            p, r, k = route_capped(G, a3, b3, pref)
            s = set(zip(p, p[1:]))
            ov = len(s & base_set) / len(s) * 100
            print(f"  {nm:13s} {r['km']:5.1f}km {r['detour']*100:+5.0f}% "
                  f"{r['fun']:6.3f} {r['tree']:6.3f} {r['green']:6.3f} "
                  f"{r['water']:6.3f} {r['smooth']:7.3f} {ov:5.1f}% {k:5.1f}")

            
# ---------- 14. 실제 구간 데모 ----------
print("\n" + "=" * 70)
print("실제 구간 경로 데모")
print("=" * 70)

DEMOS = [
    ("북악 일대",   "경복궁",      37.5796, 126.9770, "정릉",        37.6100, 127.0100),
    ("강남 일대",   "강남역",      37.4979, 127.0276, "양재시민의숲", 37.4705, 127.0380),
    ("한강 북단",   "여의도공원",   37.5265, 126.9242, "반포한강공원",  37.5100, 126.9960),
    ("광진·성동",  "건대입구역",   37.5405, 127.0700, "서울숲",      37.5444, 127.0374),
]

# 연결 요소 안의 노드만 후보로
nd_ll   = nodes_g.set_index("id")[["lon", "lat"]]
nd_big  = nd_ll[nd_ll.index.isin(big)]
arr_big = nd_big.values
ids_big = nd_big.index.values
print(f"후보 노드 {len(nd_big):,}개")

def nearest_node(lat, lon):
    d = (arr_big[:, 0] - lon) ** 2 + (arr_big[:, 1] - lat) ** 2
    return int(ids_big[int(np.argmin(d))])

def road_sequence(G, path, min_m=200):
    """경로가 지나는 도로명을 순서대로 (일정 길이 이상만)"""
    seq, cur, acc = [], None, 0.0
    for x, y in zip(path, path[1:]):
        nm = G[x][y].get("name", "") or "(이름없음)"
        L = G[x][y]["length"]
        if nm == cur:
            acc += L
        else:
            if cur is not None and acc >= min_m:
                seq.append((cur, acc))
            cur, acc = nm, L
    if cur is not None and acc >= min_m:
        seq.append((cur, acc))
    return seq

for region, n1, la1, lo1, n2, la2, lo2 in DEMOS:
    a = nearest_node(la1, lo1)
    b = nearest_node(la2, lo2)
    try:
        base = nx.shortest_path_length(G, a, b, weight="length")
    except nx.NetworkXNoPath:
        print(f"\n[{region}] {n1} → {n2}: 경로 없음 (일방통행·단절) — 건너뜀")
        continue

    print(f"\n{'─'*70}")
    print(f"[{region}]  {n1} → {n2}")
    print(f"  최단거리 {base/1000:.1f}km")

    base_p, base_r = route(G, a, b, {}, base)
    base_set = set(zip(base_p, base_p[1:]))

    for nm, pref in PRESETS.items():
        try:
            p, r, k = route_capped(G, a, b, pref)
        except nx.NetworkXNoPath:
            print(f"\n  ▸ {nm}: 경로 없음")
            continue
        s  = set(zip(p, p[1:]))
        ov = len(s & base_set) / len(s) * 100

        print(f"\n  ▸ {nm}")
        print(f"    거리 {r['km']:.1f}km  (최단 대비 {r['detour']*100:+.0f}%, "
              f"{r['km'] - base/1000:+.1f}km)  최단경로와 {ov:.0f}% 동일")

        if nm != "최단":
            chg = []
            for key, lab in [("green", "녹지"), ("tree", "가로수"),
                             ("water", "수변"), ("smooth", "소통"),
                             ("fun", "종합")]:
                b0, b1 = base_r[key], r[key]
                if abs(b1 - b0) > 0.01:
                    chg.append(f"{lab} {b0:.2f}→{b1:.2f}")
            print(f"    변화: {' / '.join(chg) if chg else '최단경로와 동일'}")

        roads = road_sequence(G, p)
        if roads:
            txt = " → ".join(f"{rn}({L/1000:.1f}km)" for rn, L in roads[:8])
            if len(roads) > 8:
                txt += f" → … (외 {len(roads)-8}개)"
            print(f"    경로: {txt}")

###오류 확인용 코드
a = nearest_node(37.5405, 127.0700)   # 건대입구
b = nearest_node(37.5444, 127.0374)   # 서울숲
print("출발 노드", a, "도착 노드", b)

und = G.to_undirected()
comps = {n: i for i, c in enumerate(nx.connected_components(und)) for n in c}
print("출발 요소", comps.get(a), "/ 도착 요소", comps.get(b))

# 무방향으로는 이어지는지
print("무방향 경로:", nx.has_path(und, a, b))

print(seg["oneway"].value_counts(dropna=False))
print("일방통행 비율:", 
      seg["oneway"].astype(str).str.lower().isin(["yes","true","1"]).mean())
