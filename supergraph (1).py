"""
================================================================================
SUPERGRAPH - HỆ THỐNG ĐỒ THỊ & TỐI ƯU LỘ TRÌNH SIÊU THỊ
================================================================================
Đáp ứng đầy đủ các yêu cầu:
1. Input đồ thị (có hướng / vô hướng)
2. Vẽ & lưu hình đồ thị (hiển thị + file PNG)
3. Biểu diễn: ma trận kề, danh sách kề, danh sách cạnh
4. Duyệt BFS, DFS
5. Kiểm tra đồ thị hai phía (bipartite)
6. Đường đi ngắn nhất: Dijkstra, Bellman-Ford
7. Thuật toán nâng cao (cài đặt đầy đủ):
   7.1 Fleury  7.2 Hierholzer  7.3 Prim  7.4 Kruskal  7.5 Ford-Fulkerson
8. Bài toán thực tế: Tối ưu lộ trình lấy hàng siêu thị
================================================================================
"""

import heapq
import itertools
import copy
from collections import defaultdict, deque
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


# ============================================================
#  LỚP ĐỒ THỊ
# ============================================================
class Graph:
    def __init__(self, directed=False):
        self.directed = directed
        self.adj   = defaultdict(dict)   # {u: {v: weight}}
        self.vertices = set()
        self.edges = []                  # list of (u, v, weight)

    # ------ thêm đỉnh / cạnh ------
    def add_vertex(self, v):
        self.vertices.add(v)
        if v not in self.adj:
            self.adj[v] = {}

    def add_edge(self, u, v, w=1.0):
        self.vertices.add(u); self.vertices.add(v)
        if v not in self.adj[u]:          # tránh trùng trong edges
            self.edges.append((u, v, w))
        self.adj[u][v] = w
        if not self.directed:
            self.adj[v][u] = w

    # ------ copy nội bộ ------
    def clone(self):
        g = Graph(self.directed)
        g.vertices = set(self.vertices)
        g.edges    = list(self.edges)
        g.adj      = defaultdict(dict, {u: dict(nb) for u, nb in self.adj.items()})
        return g

    def remove_edge(self, u, v):
        """Xoá cạnh u-v (và v-u nếu vô hướng). Không xoá khỏi self.edges."""
        if v in self.adj[u]:
            del self.adj[u][v]
        if not self.directed and u in self.adj[v]:
            del self.adj[v][u]

    def degree(self, v):
        if self.directed:
            out = len(self.adj[v])
            in_ = sum(1 for u in self.vertices if v in self.adj[u])
            return out, in_
        return len(self.adj[v])

    # ------ kiểm tra liên thông (bỏ qua đỉnh cô lập) ------
    def is_connected(self):
        non_isolated = [v for v in self.vertices if self.adj[v]]
        if not non_isolated:
            return True
        visited = set()
        q = deque([non_isolated[0]])
        while q:
            u = q.popleft()
            if u in visited: continue
            visited.add(u)
            for nb in self.adj[u]:
                q.append(nb)
            if not self.directed:
                for v2 in self.vertices:
                    if u in self.adj[v2] and v2 not in visited:
                        q.append(v2)
        return all(v in visited for v in non_isolated)


# ============================================================
#  PHẦN CƠ BẢN
# ============================================================

# ---------- 3. BIỂU DIỄN ----------
def show_representations(g):
    print("\n" + "─"*60)
    print("  DANH SÁCH CẠNH (Edge List)")
    print("─"*60)
    for u, v, w in g.edges:
        arrow = "→" if g.directed else "↔"
        print(f"  {u} {arrow} {v}  :  {w}")

    vertices = sorted(g.vertices)
    idx = {v: i for i, v in enumerate(vertices)}
    n   = len(vertices)

    print("\n" + "─"*60)
    print("  MA TRẬN KỀ (Adjacency Matrix)")
    print("─"*60)
    header = "     " + "  ".join(f"{v:>4}" for v in vertices)
    print(header)
    mat = [[0]*n for _ in range(n)]
    for u, v, w in g.edges:
        mat[idx[u]][idx[v]] = w
        if not g.directed:
            mat[idx[v]][idx[u]] = w
    for i, row in enumerate(mat):
        row_s = "  ".join(f"{x:>4}" if x != int(x) else f"{int(x):>4}" for x in row)
        print(f"  {vertices[i]:>3}  {row_s}")

    print("\n" + "─"*60)
    print("  DANH SÁCH KỀ (Adjacency List)")
    print("─"*60)
    for v in vertices:
        neigh = g.adj[v]
        s = ", ".join(f"{nb}({w})" for nb, w in neigh.items()) if neigh else "(không có)"
        print(f"  {v}  →  [{s}]")


# ---------- 4. BFS & DFS ----------
def bfs(graph, start):
    if start not in graph.vertices: return []
    visited, q, order = set(), deque([start]), []
    while q:
        u = q.popleft()
        if u in visited: continue
        visited.add(u); order.append(u)
        for v in graph.adj[u]:
            if v not in visited: q.append(v)
    return order

def dfs(graph, start):
    if start not in graph.vertices: return []
    visited, order = set(), []
    def _dfs(v):
        visited.add(v); order.append(v)
        for nb in graph.adj[v]:
            if nb not in visited: _dfs(nb)
    _dfs(start)
    return order


# ---------- 5. BIPARTITE ----------
def is_bipartite(graph):
    und = Graph(directed=False)
    for u, v, w in graph.edges:
        und.add_edge(u, v, w)
    color = {}
    for v in und.vertices:
        if v not in color:
            q = deque([v]); color[v] = 0
            while q:
                u = q.popleft()
                for nb in und.adj[u]:
                    if nb not in color:
                        color[nb] = 1 - color[u]; q.append(nb)
                    elif color[nb] == color[u]:
                        return False, {}
    # tách 2 tập
    set0 = [v for v, c in color.items() if c == 0]
    set1 = [v for v, c in color.items() if c == 1]
    return True, {"Tập U": set0, "Tập V": set1}


# ---------- 6. DIJKSTRA & BELLMAN-FORD ----------
def dijkstra(graph, start, end):
    if start not in graph.vertices or end not in graph.vertices:
        return None, float('inf')
    dist = {v: float('inf') for v in graph.vertices}
    prev = {v: None for v in graph.vertices}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        if u == end: break
        for v, w in graph.adj[u].items():
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd; prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dist[end] == float('inf'): return None, float('inf')
    path, cur = [], end
    while cur is not None:
        path.append(cur); cur = prev[cur]
    return list(reversed(path)), dist[end]

def bellman_ford(graph, start, end):
    if start not in graph.vertices or end not in graph.vertices:
        return None, float('inf'), False
    vertices = list(graph.vertices)
    dist = {v: float('inf') for v in vertices}
    prev = {v: None for v in vertices}
    dist[start] = 0
    edges = list(graph.edges)
    if not graph.directed:
        edges += [(v, u, w) for u, v, w in graph.edges]
    for _ in range(len(vertices) - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w; prev[v] = u; updated = True
        if not updated: break
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return None, float('inf'), True          # chu trình âm
    if dist[end] == float('inf'): return None, float('inf'), False
    path, cur = [], end
    while cur is not None:
        path.append(cur); cur = prev[cur]
    return list(reversed(path)), dist[end], False


# ============================================================
#  PHẦN NÂNG CAO — cài đặt đầy đủ
# ============================================================

# ---- 7.1 FLEURY ----
def _has_euler_path_or_circuit(graph):
    """
    Kiểm tra tồn tại đường / chu trình Euler.
    Trả về ('circuit', start) hoặc ('path', start) hoặc (None, None).
    """
    if not graph.is_connected():
        return None, None
    if graph.directed:
        odd_out, odd_in = [], []
        for v in graph.vertices:
            out = len(graph.adj[v])
            in_ = sum(1 for u in graph.vertices if v in graph.adj[u])
            if out - in_ == 1:  odd_out.append(v)
            elif in_ - out == 1: odd_in.append(v)
            elif out != in_:
                return None, None
        if len(odd_out) == 0 and len(odd_in) == 0:
            start = next(iter(graph.vertices))
            return 'circuit', start
        if len(odd_out) == 1 and len(odd_in) == 1:
            return 'path', odd_out[0]
        return None, None
    else:
        odd = [v for v in graph.vertices if len(graph.adj[v]) % 2 == 1]
        if len(odd) == 0:
            return 'circuit', next(iter(graph.vertices))
        if len(odd) == 2:
            return 'path', odd[0]
        return None, None

def _is_bridge(g, u, v):
    """Kiểm tra (u,v) có phải cầu không bằng cách so sánh số đỉnh có thể tiếp cận trước và sau khi xóa cạnh."""
    reachable_before = len(bfs(g, u))
    g2 = g.clone()
    g2.remove_edge(u, v)
    reachable_after = len(bfs(g2, u))
    return reachable_after < reachable_before

def fleury(graph):
    """
    Thuật toán Fleury — tìm đường / chu trình Euler.
    Trả về (kiểu, danh sách đỉnh) hoặc (None, []).
    """
    kind, start = _has_euler_path_or_circuit(graph)
    if kind is None:
        return None, []
    g = graph.clone()
    path = [start]
    cur  = start
    while True:
        neighbors = list(g.adj[cur].keys())
        if not neighbors:
            break
        chosen = None
        for nb in neighbors:
            if not _is_bridge(g, cur, nb):
                chosen = nb; break
        if chosen is None:
            chosen = neighbors[0]
        path.append(chosen)
        g.remove_edge(cur, chosen)
        cur = chosen
    return kind, path


# ---- 7.2 HIERHOLZER ----
def hierholzer(graph):
    """
    Thuật toán Hierholzer O(E) — tìm chu trình / đường Euler.
    Trả về (kiểu, danh sách đỉnh) hoặc (None, []).
    """
    kind, start = _has_euler_path_or_circuit(graph)
    if kind is None:
        return None, []
    # làm việc trên bản sao adj
    adj = {v: list(nb.keys()) for v, nb in graph.adj.items()}
    # dùng stack
    stack  = [start]
    circuit = []
    while stack:
        v = stack[-1]
        if adj[v]:
            nb = adj[v].pop()
            # xoá chiều ngược nếu vô hướng
            if not graph.directed and v in adj.get(nb, []):
                adj[nb].remove(v)
            stack.append(nb)
        else:
            circuit.append(stack.pop())
    circuit.reverse()
    return kind, circuit


# ---- 7.3 PRIM ----
def prim_mst(graph):
    if graph.directed:
        raise ValueError("Prim chỉ áp dụng cho đồ thị vô hướng.")
    if not graph.vertices:
        return 0, []
    start   = next(iter(graph.vertices))
    visited = {start}
    heap    = [(w, start, nb) for nb, w in graph.adj[start].items()]
    heapq.heapify(heap)
    mst_cost, mst_edges = 0, []
    while heap and len(visited) < len(graph.vertices):
        w, u, v = heapq.heappop(heap)
        if v in visited: continue
        visited.add(v)
        mst_cost += w
        mst_edges.append((u, v, w))
        for nb, w2 in graph.adj[v].items():
            if nb not in visited:
                heapq.heappush(heap, (w2, v, nb))
    if len(visited) < len(graph.vertices):
        raise ValueError("Đồ thị không liên thông — không có MST.")
    return mst_cost, mst_edges


# ---- 7.4 KRUSKAL ----
def kruskal_mst(graph):
    if graph.directed:
        raise ValueError("Kruskal chỉ áp dụng cho đồ thị vô hướng.")
    parent = {v: v for v in graph.vertices}
    rank   = {v: 0 for v in graph.vertices}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]; v = parent[v]
        return v

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb: return False
        if rank[ra] < rank[rb]: parent[ra] = rb
        elif rank[ra] > rank[rb]: parent[rb] = ra
        else: parent[rb] = ra; rank[ra] += 1
        return True

    sorted_edges = sorted(graph.edges, key=lambda x: x[2])
    mst_cost, mst_edges = 0, []
    for u, v, w in sorted_edges:
        if union(u, v):
            mst_cost += w; mst_edges.append((u, v, w))
    return mst_cost, mst_edges


# ---- 7.5 FORD-FULKERSON (BFS / Edmonds-Karp) ----
def ford_fulkerson(graph, source, sink):
    """
    Ford-Fulkerson dùng BFS (Edmonds-Karp) — O(VE^2).
    Trả về (max_flow, flow_dict).
    flow_dict[(u,v)] = lượng luồng trên cạnh u→v.
    """
    if source not in graph.vertices or sink not in graph.vertices:
        return 0, {}

    # Xây dựng đồ thị khả năng (capacity graph) — đồng thời xử lý
    # đồ thị vô hướng bằng cách thêm 2 chiều
    cap = defaultdict(lambda: defaultdict(float))
    for u, v, w in graph.edges:
        cap[u][v] += w
        if not graph.directed:
            cap[v][u] += w

    # Luồng hiện tại
    flow = defaultdict(lambda: defaultdict(float))
    all_nodes = list(graph.vertices)

    def bfs_path():
        """BFS tìm đường tăng trong đồ thị thặng dư."""
        parent = {source: None}
        q = deque([source])
        while q:
            u = q.popleft()
            for v in all_nodes:
                residual = cap[u][v] - flow[u][v]
                if v not in parent and residual > 0:
                    parent[v] = u
                    if v == sink:
                        return parent
                    q.append(v)
        return None

    max_flow = 0.0
    while True:
        parent = bfs_path()
        if parent is None:
            break
        # Tìm bottleneck
        path_flow = float('inf')
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, cap[u][v] - flow[u][v])
            v = u
        # Cập nhật luồng
        v = sink
        while v != source:
            u = parent[v]
            flow[u][v] += path_flow
            flow[v][u] -= path_flow
            v = u
        max_flow += path_flow

    # Trả về flow thực dương
    flow_dict = {(u, v): flow[u][v] for u in flow for v in flow[u] if flow[u][v] > 0}
    return max_flow, flow_dict


# ============================================================
#  BÀI TOÁN THỰC TẾ (yêu cầu 8)
# ============================================================
def optimize_supermarket_route(graph, start, must_pick, end=None):
    """
    Dijkstra + brute-force permutation để tìm lộ trình ngắn nhất
    qua tất cả kệ hàng bắt buộc (must_pick).
    """
    if end is None:
        end = start
    valid = [v for v in must_pick if v in graph.vertices]
    if not valid:
        return dijkstra(graph, start, end)

    nodes = list(dict.fromkeys([start] + valid + [end]))
    dist_mat, path_mat = {}, {}
    for u in nodes:
        for v in nodes:
            if u == v:
                dist_mat[(u, v)] = 0; path_mat[(u, v)] = [u]
            else:
                p, d = dijkstra(graph, u, v)
                dist_mat[(u, v)] = d; path_mat[(u, v)] = p or []

    best_cost, best_seq = float('inf'), None
    for perm in itertools.permutations(valid):
        seq  = [start] + list(perm) + ([end] if end != start else [])
        cost = sum(dist_mat[(seq[i], seq[i+1])] for i in range(len(seq)-1))
        if cost < best_cost:
            best_cost = cost; best_seq = seq

    if best_seq is None:
        return None, float('inf')

    full_path = [start]
    for i in range(len(best_seq)-1):
        seg = path_mat[(best_seq[i], best_seq[i+1])]
        full_path.extend(seg[1:])
    return full_path, best_cost


# ============================================================
#  VẼ & LƯU ĐỒ THỊ
# ============================================================
def _build_nx(graph):
    G = nx.DiGraph() if graph.directed else nx.Graph()
    for u, v, w in graph.edges:
        G.add_edge(u, v, weight=w)
    for v in graph.vertices:
        if v not in G:
            G.add_node(v)
    return G

def _draw(graph, G, pos, highlight_edges=None, highlight_nodes=None,
          title="Đồ thị", flow_dict=None, mst_edges=None):
    plt.figure(figsize=(9, 7))
    ax = plt.gca()

    # Màu node
    node_colors = []
    for n in G.nodes():
        if highlight_nodes and n in highlight_nodes:
            node_colors.append('#ff6b6b')
        else:
            node_colors.append('#74b9ff')
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    # Tất cả cạnh
    edge_colors, edge_widths = [], []
    all_edges = list(G.edges())
    mst_set = set()
    if mst_edges:
        mst_set = {(u, v) for u, v, _ in mst_edges} | {(v, u) for u, v, _ in mst_edges}
    hl_set = set()
    if highlight_edges:
        hl_set = set(map(tuple, highlight_edges))

    for e in all_edges:
        if e in hl_set or (e[1], e[0]) in hl_set:
            edge_colors.append('#d63031'); edge_widths.append(4.0)
        elif e in mst_set or (e[1], e[0]) in mst_set:
            edge_colors.append('#00b894'); edge_widths.append(3.5)
        else:
            edge_colors.append('#b2bec3'); edge_widths.append(1.5)

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                           alpha=0.85, ax=ax,
                           arrows=graph.directed,
                           arrowstyle='-|>', arrowsize=20,
                           connectionstyle='arc3,rad=0.07' if graph.directed else 'arc3,rad=0')

    # Nhãn cạnh
    if flow_dict:
        # Hiển thị luồng / khả năng
        edge_labels = {}
        cap_map = {(u, v): w for u, v, w in graph.edges}
        if not graph.directed:
            cap_map.update({(v, u): w for u, v, w in graph.edges})
        for u, v, w in graph.edges:
            f = flow_dict.get((u, v), 0)
            edge_labels[(u, v)] = f"{f}/{w}"
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    else:
        edge_labels = {(u, v): f"{w}" for u, v, w in graph.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    plt.title(title, fontsize=13, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

def draw_graph(graph, highlight_path=None, highlight_nodes=None,
               title="Sơ đồ đồ thị", flow_dict=None, mst_edges=None):
    G   = _build_nx(graph)
    pos = nx.spring_layout(G, seed=42)
    hl_edges = None
    if highlight_path and len(highlight_path) > 1:
        hl_edges = [(highlight_path[i], highlight_path[i+1])
                    for i in range(len(highlight_path)-1)]
    _draw(graph, G, pos, hl_edges, highlight_nodes, title, flow_dict, mst_edges)
    plt.show()

def save_graph_image(graph, filename, highlight_path=None, highlight_nodes=None,
                     title="Sơ đồ đồ thị", flow_dict=None, mst_edges=None):
    G   = _build_nx(graph)
    pos = nx.spring_layout(G, seed=42)
    hl_edges = None
    if highlight_path and len(highlight_path) > 1:
        hl_edges = [(highlight_path[i], highlight_path[i+1])
                    for i in range(len(highlight_path)-1)]
    _draw(graph, G, pos, hl_edges, highlight_nodes, title, flow_dict, mst_edges)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu ảnh: {filename}")


# ============================================================
#  GIAO DIỆN CONSOLE
# ============================================================
def print_menu():
    print("\n" + "="*70)
    print("  SUPERGRAPH — ĐỒ THỊ & TỐI ƯU LỘ TRÌNH".center(70))
    print("="*70)
    print(" [1]  Nhập đồ thị")
    print(" [2]  Hiển thị biểu diễn đồ thị")
    print(" [3]  Duyệt BFS / DFS")
    print(" [4]  Kiểm tra đồ thị hai phía (bipartite)")
    print(" [5]  Đường đi ngắn nhất  (Dijkstra & Bellman-Ford)")
    print(" [6]  Vẽ đồ thị (cửa sổ)")
    print(" [7]  Lưu ảnh đồ thị PNG")
    print(" [8]  Thuật toán nâng cao")
    print("        8a. Fleury        8b. Hierholzer")
    print("        8c. Prim          8d. Kruskal")
    print("        8e. Ford-Fulkerson")
    print(" [9]  ★ Bài toán thực tế: Tối ưu lộ trình siêu thị")
    print(" [0]  Thoát")
    print("─"*70)

def _prompt(msg, default=None):
    s = input(msg).strip()
    return s if s else default

def input_graph():
    print("\nChọn nguồn dữ liệu:")
    print(" 1. Nhập thủ công")
    print(" 2. Đồ thị mẫu vô hướng")
    print(" 3. Đồ thị mẫu có hướng")
    print(" 4. 📂 Load từ file .txt")   # 👈 thêm dòng này

    ch = _prompt("Lựa chọn [1/2/3/4]: ", '1')

    # --- LOAD FILE ---
    if ch == '4':
        filename = _prompt("Nhập tên file (vd: test.txt): ", '')
        directed_flag = _prompt("Đồ thị có hướng? (y/n): ", 'n')

        return load_graph_from_file(filename, directed_flag)

    # --- phần cũ giữ nguyên ---
    elif ch == '2':
        g = Graph(directed=False)
        for u, v, w in [("A","B",2),("A","C",4),("B","C",1)]:
            g.add_edge(u, v, w)
        return g

    elif ch == '3':
        g = Graph(directed=True)
        for u, v, w in [("S","A",10),("A","T",5)]:
            g.add_edge(u, v, w)
        return g

    else:
        directed = _prompt("Đồ thị có hướng? (y/n): ", 'n').lower() == 'y'
        g = Graph(directed)

        print("Nhập cạnh: u v w")
        while True:
            line = input("> ").strip()
            if not line:
                break

            parts = line.split()
            u, v = parts[0], parts[1]
            w = float(parts[2]) if len(parts) >= 3 else 1.0
            g.add_edge(u, v, w)

        return g
# ============================================================
#  MENU CHÍNH
# ============================================================
def main():
    graph = None

    while True:
        print_menu()
        cmd = _prompt("Nhập lệnh: ", '').lower()

        if cmd == '0':
            print("Tạm biệt!")
            break

        # ---- 1. Nhập đồ thị ----
        elif cmd == '1':
            graph = input_graph()
            print(f"✅ Đồ thị {'có hướng' if graph.directed else 'vô hướng'} — "
                  f"{len(graph.vertices)} đỉnh, {len(graph.edges)} cạnh.")

        # ---- 2. Biểu diễn ----
        elif cmd == '2':
            if graph is None: print("❌ Chưa có đồ thị."); continue
            show_representations(graph)

        # ---- 3. BFS / DFS ----
        elif cmd == '3':
            if graph is None: print("❌ Chưa có đồ thị."); continue
            start = _prompt("Đỉnh bắt đầu: ", '')
            if start not in graph.vertices:
                print("❌ Đỉnh không tồn tại."); continue
            b = bfs(graph, start); d = dfs(graph, start)
            print(f"BFS: {' → '.join(b)}")
            print(f"DFS: {' → '.join(d)}")
            if _prompt("Vẽ minh hoạ? (y/n): ", 'n') == 'y':
                draw_graph(graph, highlight_nodes=set(b),
                           title=f"BFS/DFS từ {start}")

        # ---- 4. Bipartite ----
        elif cmd == '4':
            if graph is None: print("❌ Chưa có đồ thị."); continue
            ok, sets = is_bipartite(graph)
            if ok:
                print("✅ Đồ thị LÀ hai phía.")
                for k, v in sets.items(): print(f"   {k}: {v}")
            else:
                print("❌ Đồ thị KHÔNG phải hai phía.")

        # ---- 5. Shortest path ----
        elif cmd == '5':
            if graph is None: print("❌ Chưa có đồ thị."); continue
            s = _prompt("Đỉnh xuất phát: ", '')
            t = _prompt("Đỉnh đích    : ", '')
            # Dijkstra
            path, dist_ = dijkstra(graph, s, t)
            if path:
                print(f"  Dijkstra    : {' → '.join(path)}  |  chi phí = {dist_}")
            else:
                print("  Dijkstra    : không có đường.")
            # Bellman-Ford
            path2, dist2, neg = bellman_ford(graph, s, t)
            if neg:
                print("  Bellman-Ford: phát hiện chu trình âm!")
            elif path2:
                print(f"  Bellman-Ford: {' → '.join(path2)}  |  chi phí = {dist2}")
            else:
                print("  Bellman-Ford: không có đường.")
            if path and _prompt("Vẽ đường đi? (y/n): ", 'n') == 'y':
                draw_graph(graph, highlight_path=path,
                           title=f"Đường ngắn nhất {s}→{t}")

        # ---- 6. Vẽ đồ thị ----
        elif cmd == '6':
            if graph is None: print("❌ Chưa có đồ thị."); continue
            draw_graph(graph)

        # ---- 7. Lưu PNG ----
        elif cmd == '7':
            if graph is None: print("❌ Chưa có đồ thị."); continue
            fname = _prompt("Tên file PNG (mặc định graph.png): ", 'graph.png')
            save_graph_image(graph, fname)

        # ---- 8. Thuật toán nâng cao ----
        elif cmd == '8':
            if graph is None: print("❌ Chưa có đồ thị."); continue
            sub = _prompt("Chọn thuật toán [a/b/c/d/e]: ", '').lower()

            # -- 8a Fleury --
            if sub == 'a':
                kind, path = fleury(graph)
                if kind is None:
                    print("❌ Đồ thị không có đường/chu trình Euler.")
                else:
                    print(f"✅ Fleury tìm được {'chu trình' if kind=='circuit' else 'đường'} Euler:")
                    print("   " + " → ".join(path))
                    if _prompt("Vẽ? (y/n): ", 'n') == 'y':
                        draw_graph(graph, highlight_path=path,
                                   title="Fleury — Euler " + ("Circuit" if kind=='circuit' else "Path"))

            # -- 8b Hierholzer --
            elif sub == 'b':
                kind, path = hierholzer(graph)
                if kind is None:
                    print("❌ Đồ thị không có đường/chu trình Euler.")
                else:
                    print(f"✅ Hierholzer tìm được {'chu trình' if kind=='circuit' else 'đường'} Euler:")
                    print("   " + " → ".join(path))
                    if _prompt("Vẽ? (y/n): ", 'n') == 'y':
                        draw_graph(graph, highlight_path=path,
                                   title="Hierholzer — Euler " + ("Circuit" if kind=='circuit' else "Path"))

            # -- 8c Prim --
            elif sub == 'c':
                try:
                    cost, edges = prim_mst(graph)
                    print(f"✅ Prim MST — tổng trọng số: {cost}")
                    for e in edges: print(f"   {e[0]} — {e[1]}  ({e[2]})")
                    if _prompt("Vẽ MST? (y/n): ", 'n') == 'y':
                        draw_graph(graph, mst_edges=edges, title="Prim — Cây khung nhỏ nhất")
                except ValueError as e:
                    print(f"❌ {e}")

            # -- 8d Kruskal --
            elif sub == 'd':
                try:
                    cost, edges = kruskal_mst(graph)
                    print(f"✅ Kruskal MST — tổng trọng số: {cost}")
                    for e in edges: print(f"   {e[0]} — {e[1]}  ({e[2]})")
                    if _prompt("Vẽ MST? (y/n): ", 'n') == 'y':
                        draw_graph(graph, mst_edges=edges, title="Kruskal — Cây khung nhỏ nhất")
                except ValueError as e:
                    print(f"❌ {e}")

            # -- 8e Ford-Fulkerson --
            elif sub == 'e':
                if not graph.directed:
                    print("⚠️  Đồ thị vô hướng — Ford-Fulkerson vẫn chạy được (coi mỗi cạnh 2 chiều).")
                src  = _prompt("Đỉnh nguồn (source): ", 'S')
                sink = _prompt("Đỉnh đích  (sink)  : ", 'T')
                mf, fdict = ford_fulkerson(graph, src, sink)
                print(f"✅ Luồng cực đại từ {src} → {sink}: {mf}")
                for (u, v), f in sorted(fdict.items()):
                    cap_val = graph.adj[u].get(v, 0)
                    print(f"   {u} → {v}: {f}/{cap_val}")
                if _prompt("Vẽ luồng? (y/n): ", 'n') == 'y':
                    draw_graph(graph, flow_dict=fdict,
                               highlight_nodes={src, sink},
                               title=f"Ford-Fulkerson — max flow = {mf}")
            else:
                print("❌ Chọn a/b/c/d/e.")

        # ---- 9. Bài toán thực tế ----
        elif cmd == '9':
            if graph is None: print("❌ Chưa có đồ thị."); continue
            start = _prompt("Đỉnh xuất phát (mặc định A): ", 'A')
            picks_str = _prompt("Các kệ bắt buộc (VD: C,E,F): ", '')
            if not picks_str: print("❌ Chưa nhập kệ."); continue
            picks = [p.strip() for p in picks_str.split(',') if p.strip()]
            end   = _prompt(f"Đỉnh kết thúc (mặc định quay lại {start}): ", start)
            path, cost = optimize_supermarket_route(graph, start, picks, end)
            if path is None:
                print("❌ Không tìm được lộ trình.")
            else:
                print(f"\n🛒 LỘ TRÌNH TỐI ƯU : {' → '.join(path)}")
                print(f"📏 TỔNG QUÃNG ĐƯỜNG: {cost}")
                if _prompt("Vẽ lộ trình? (y/n): ", 'n') == 'y':
                    draw_graph(graph, highlight_path=path,
                               title="Lộ trình tối ưu siêu thị")

        else:
            print("❌ Lệnh không hợp lệ. Chọn 0-9.")

import sys

def load_graph_from_file(filename, directed_flag):
    directed = directed_flag.lower() == 'y'
    g = Graph(directed)

    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    u, v = parts[0], parts[1]
                    w = float(parts[2]) if len(parts) >= 3 else 1.0
                    g.add_edge(u, v, w)

        return g

    except Exception as e:
        print("❌ Lỗi đọc file:", e)
        return None

if __name__ == "__main__":
    main()