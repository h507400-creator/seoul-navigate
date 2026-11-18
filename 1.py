# 1.py — 경로 탐색 임시 버전 (프론트엔드·DB 연동 테스트용)
#   seoul_segments_final.gpkg / seoul_nodes.gpkg / restrictions.csv 만 있으면 동작 (같은 폴더)
#   find_route(출발위도, 출발경도, 도착위도, 도착경도, 프리셋) → 지도에 그릴 (위도, 경도) 좌표 포함 결과
import os, json, time
import numpy as np, pandas as pd, geopandas as gpd, networkx as nx

BASE = os.path.dirname(os.path.abspath(__file__))
MAX_DETOUR = 0.30                      # 최단거리 대비 우회 허용 한도 (30%)
PRESETS = {
    "최단":         {},
    "숲길 드라이브":  {"fun": 1},
    "가로수길":      {"tree": 1},
    "공원길":       {"green": 1},
    "강변길":       {"water": 1},
    "안 막히는 길":   {"smooth": 1},
}

# ====================================================================== 1. 그래프 복원
t0 = time.time()
seg = gpd.read_file(os.path.join(BASE, "seoul_segments_final.gpkg")).to_crs(4326)
nodes = gpd.read_file(os.path.join(BASE, "seoul_nodes.gpkg"), ignore_geometry=True)
rst = pd.read_csv(os.path.join(BASE, "restrictions.csv"))
LAT = dict(zip(nodes["id"], nodes["lat"])); LON = dict(zip(nodes["id"], nodes["lon"]))

G = nx.DiGraph()
def add_short(u, v, a):
    """같은 두 교차로 사이에 도로가 여러 개면 짧은 쪽만 남김"""
    if G.has_edge(u, v) and G[u][v]["length"] <= a["length"]:
        return
    G.add_edge(u, v, **a)

for r in seg.itertuples(index=False):
    ow = str(r.oneway).strip().lower()
    hw, jct = str(r.highway), str(r.junction)
    rev = ow == "-1"
    one = ow in ("yes", "true", "1") or rev
    # oneway 태그가 비어 있어도 고속도로·램프·회전교차로는 일방통행 (OSM 규칙)
    if ow in ("nan", "none", "") and (hw in ("motorway", "motorway_link", "trunk_link")
                                      or jct in ("roundabout", "circular")):
        one = True
    coords = [(round(y, 7), round(x, 7)) for x, y in r.geometry.coords]   # (위도, 경도), u→v 순서
    a = dict(length=r.length, way=int(r.id), name=r.road_name if isinstance(r.road_name, str) else "",
             speed=r.speed_est, expwy=bool(r.is_expwy), tunnel=bool(r.is_tunnel),
             fun=r.fun, tree=r.tree_s, green=r.green_s, water=r.water_s, smooth=r.smooth_s,
             coords=coords)
    u, v = int(r.u), int(r.v)
    back = dict(a, coords=coords[::-1])                                  # v→u 방향은 좌표도 뒤집음
    if rev:
        add_short(v, u, back)
    else:
        add_short(u, v, a)
        if not one:
            add_short(v, u, back)

# ---- 회전 제한: 제한이 있는 교차로 y만 "어디서 들어왔는지"별로 쪼갬 ----
#  x에서 y로 들어온 차는 가상 노드 ("t", y, x)를 거쳐서만 나갈 수 있고,
#  금지된 회전(예: 좌회전 금지) 방향으로는 엣지를 만들지 않는다.
#  ("t", y, x)는 실제로는 교차로 y와 같은 자리이므로, 결과를 낼 때 y로 되돌린다 (real_node).
NO = {(int(f), int(v), int(t)) for k, f, v, t in rst.itertuples(index=False) if k.startswith("no_")}
ONLY = {}
for k, f, v, t in rst.itertuples(index=False):
    if k.startswith("only_"):
        ONLY.setdefault((int(f), int(v)), set()).add(int(t))
def turn_ok(w_in, y, w_out):
    if (w_in, y, w_out) in NO: return False
    if (w_in, y) in ONLY and w_out not in ONLY[(w_in, y)]: return False
    return True
VIA = ({v for _, v, _ in NO} | {v for _, v in ONLY}) & set(G.nodes)
for y in VIA:
    ins, outs = list(G.in_edges(y, data=True)), list(G.out_edges(y, data=True))
    for x, _, din in ins:
        vy = ("t", y, x)
        G.add_edge(x, vy, **din)
        for _, z, dout in outs:
            if turn_ok(din["way"], y, dout["way"]):
                G.add_edge(vy, z, **dout)
        G.remove_edge(x, y)

def real_node(n):
    """회전 제한용 가상 노드 ("t", y, x) → 원래 교차로 y"""
    return n[1] if isinstance(n, tuple) else n

BIG = max(nx.strongly_connected_components(G), key=len)    # 서로 오갈 수 있는 범위
# 출발·도착 후보: 일반도로(고속화도로·터널 제외) 위의 실제 교차로, 회전제한 교차로는 제외
ok = set()
for x, y, d in G.edges(data=True):
    if not d["expwy"] and not d["tunnel"]:
        ok.update(n for n in (x, y) if not isinstance(n, tuple))
CAND = np.array(sorted((ok - VIA) & BIG))
CAND_LAT = np.array([LAT[n] for n in CAND]); CAND_LON = np.array([LON[n] for n in CAND])
print(f"[그래프] 노드 {G.number_of_nodes():,} / 엣지 {G.number_of_edges():,} / "
      f"회전제한 교차로 {len(VIA):,} / 출발·도착 후보 {len(CAND):,} ({time.time()-t0:.0f}초)")

# ====================================================================== 2. 경로 탐색
def nearest_node(lat, lon):
    """좌표에서 가장 가까운 출발·도착 후보 교차로"""
    d = ((CAND_LAT - lat) * 111.0) ** 2 + ((CAND_LON - lon) * 88.2) ** 2   # km² (서울 위도 기준)
    return int(CAND[int(np.argmin(d))])

def _weight(pref):
    if not pref:
        return "length"
    return lambda u, v, d: d["length"] * (1 + sum(w * (1 - d[k]) for k, w in pref.items()))

def _path(a, b, pref):
    return nx.bidirectional_dijkstra(G, a, b, weight=_weight(pref))[1]

def _plen(p):
    return sum(G[x][y]["length"] for x, y in zip(p, p[1:]))

def _capped(a, b, pref, base, iters=8):
    """우회 한도(MAX_DETOUR)를 넘지 않는 선에서 선호를 최대한 반영 (가중치 이분탐색)"""
    if not pref:
        return _path(a, b, {})
    lo, hi, best = 0.0, 50.0, None
    for _ in range(iters):
        k = (lo + hi) / 2
        p = _path(a, b, {key: w * k for key, w in pref.items()})
        if _plen(p) <= base * (1 + MAX_DETOUR):
            best, lo = p, k
        else:
            hi = k
    return best or _path(a, b, {})

def _result(p, preset, base):
    pairs = list(zip(p, p[1:]))
    coords = []
    for x, y in pairs:                                   # 엣지 도로모양을 이어붙여 한 줄로
        c = G[x][y]["coords"]
        coords.extend(c if not coords else c[1:])
    roads = []                                           # 같은 도로명 연속 구간 합치기
    for x, y in pairs:
        d = G[x][y]; nm = d["name"] or "(이름없음)"
        if roads and roads[-1]["name"] == nm:
            roads[-1]["m"] += d["length"]
        else:
            roads.append({"name": nm, "m": d["length"]})
    node_ids = []                                        # 가상 노드 → 실제 교차로, 연속 중복 제거
    for n in p:
        rn = real_node(n)
        if not node_ids or node_ids[-1] != rn:
            node_ids.append(int(rn))
    L = _plen(p)
    avg = lambda k: round(float(np.average([G[x][y][k] for x, y in pairs],
                                           weights=[G[x][y]["length"] for x, y in pairs])), 3)
    return {
        "preset": preset,
        "distance_km": round(L / 1000, 2),
        "duration_min": round(sum(G[x][y]["length"] / (G[x][y]["speed"] / 3.6) for x, y in pairs) / 60, 1),
        "detour_pct": round((L / base - 1) * 100, 1),
        "scores": {k: avg(k) for k in ("fun", "tree", "green", "water", "smooth")},
        "roads": [{"name": r["name"], "km": round(r["m"] / 1000, 2)} for r in roads],
        "coords": [[la, lo] for la, lo in coords],      # [[위도, 경도], ...] — 프론트 지도 polyline용
        "nodes": node_ids,                               # 지나가는 교차로 id (가상 노드 없음)
    }

def find_route(lat1, lon1, lat2, lon2, preset="최단"):
    """출발/도착 좌표와 프리셋 이름 → 경로 결과(dict). 프리셋은 PRESETS 키 중 하나."""
    a, b = nearest_node(lat1, lon1), nearest_node(lat2, lon2)
    base = _plen(_path(a, b, {}))
    p = _capped(a, b, PRESETS[preset], base)
    return _result(p, preset, base)

def find_all(lat1, lon1, lat2, lon2):
    """모든 프리셋 결과를 한 번에"""
    return {name: find_route(lat1, lon1, lat2, lon2, name) for name in PRESETS}

# ====================================================================== 3. 실행 예시
if __name__ == "__main__":
    O, D = (37.5405, 127.0700), (37.5444, 127.0374)        # 건대입구역 → 서울숲
    t = time.time()
    res = find_all(*O, *D)
    print(f"\n건대입구역 → 서울숲  (프리셋 {len(res)}개, {time.time()-t:.1f}초)")
    for name, r in res.items():
        top = " → ".join(f"{x['name']}({x['km']}km)" for x in r["roads"][:5])
        print(f"  {name:8s} {r['distance_km']:5.2f}km  약 {r['duration_min']:4.1f}분  "
              f"우회 {r['detour_pct']:+5.1f}%  좌표 {len(r['coords'])}개  | {top}")
    out = os.path.join(BASE, "route_example.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    print(f"\n결과 저장: {out}")
    print("좌표 예시(처음 3개, [위도, 경도]):", res["최단"]["coords"][:3])
