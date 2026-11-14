import geopandas as gpd
import networkx as nx
import os

BASE = r"C:\전전심설\hw3\team_project"

# ---------- 1. 그래프 복원 (서버 시작 시 한 번) ----------
seg = gpd.read_file(os.path.join(BASE, "seoul_segments_final.gpkg"))

G = nx.DiGraph()
for r in seg.itertuples():
    ow = str(getattr(r, "oneway", "")).strip().lower()
    rev = (ow == "-1")
    one = ow in ("yes", "true", "1") or rev

    attrs = dict(
        length = r.length,
        fun    = getattr(r, "fun_숲길", 0.5),
        fun2   = getattr(r, "fun_도심", 0.5),
        tree   = r.tree_s,
        green  = r.green_s,
        water  = r.water_s,
        culture= r.culture_s,
        smooth = r.smooth_s,
        curve  = r.curve_s,
        straight = r.straight_s,
        grade  = r.grade_s,
        rank   = r.road_rank,
        speed = r.speed_est,
        expwy  = bool(r.is_expwy),
        name   = r.road_name if isinstance(r.road_name, str) else "",
    )

    if rev:
        G.add_edge(int(r.v), int(r.u), **attrs)
    else:
        G.add_edge(int(r.u), int(r.v), **attrs)
        if not one:
            G.add_edge(int(r.v), int(r.u), **attrs)

print(f"노드 {G.number_of_nodes():,} / 엣지 {G.number_of_edges():,}")

# ---------- 2. 강한 연결 요소 (경로 탐색 가능 범위) ----------
BIG = max(nx.strongly_connected_components(G), key=len)
print(f"강한 연결 {len(BIG):,}개 ({len(BIG)/G.number_of_nodes()*100:.1f}%)")

# ---------- 3. 좌표 → 노드 ----------
import numpy as np

nodes = gpd.read_file(os.path.join(BASE, "seoul_nodes.gpkg"))
nodes = nodes[nodes["id"].isin(BIG)]
NODE_ID  = nodes["id"].values
NODE_LON = nodes["lon"].values
NODE_LAT = nodes["lat"].values

def nearest_node(lat, lon):
    d = (NODE_LON - lon) ** 2 + (NODE_LAT - lat) ** 2
    return int(NODE_ID[int(np.argmin(d))])