#!/usr/bin/env python
"""Audit hop_N namo scenes for the 'close-walled-off' shortcut property.

For each XML under external/namo-rl/scenes/car/hop_N/benchmark_3/, computes:
  E         Euclidean(car_start, goal)
  hit_dist  distance from car_start along start->goal segment to first STATIC
            wall AABB hit (inf if no wall blocks the straight line)
  G         shortest navigable path (BFS on a 1cm grid with 4cm wall padding,
            treating ONLY walls as obstacles; movables are ignored so the
            audit doesn't bake in any push assumption)
  G/E       detour ratio

Only static `wall_*` geoms count as obstacles. Movables are intentionally
treated as passable so the audit reflects the shortcut concern, not the push
dynamics that the agent is supposed to learn.
"""
import math
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

SCENE_ROOT = Path(
    "/cache/home/kb1204/code/tdmpc_square_public/external/namo-rl/scenes/car"
)
BOUNDS = (-0.70, 0.70, -0.70, 0.70)  # arena half-extent ~0.7 m
CELL = 0.01                          # 1 cm grid -> 140x140
CAR_RADIUS_PAD = 0.04                # inflate walls by ~car half-extent


def parse_scene(xml_path):
    root = ET.parse(xml_path).getroot()
    car = goal = None
    walls = []
    for body in root.iter("body"):
        if body.get("name") == "car":
            p = list(map(float, body.get("pos").split()))
            car = (p[0], p[1])
    for site in root.iter("site"):
        if site.get("name") == "goal":
            p = list(map(float, site.get("pos").split()))
            goal = (p[0], p[1])
    for geom in root.iter("geom"):
        if (geom.get("name") or "").startswith("wall_"):
            p = list(map(float, geom.get("pos").split()))
            s = list(map(float, geom.get("size").split()))
            walls.append((p[0], p[1], s[0], s[1]))
    return car, goal, walls


def first_wall_hit(start, end, walls):
    """Slab method 2D ray-AABB intersection over a segment."""
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    seg_len = math.hypot(dx, dy)
    best_t = None
    for px, py, hx, hy in walls:
        xmin, xmax = px - hx, px + hx
        ymin, ymax = py - hy, py + hy
        tmin, tmax = 0.0, 1.0
        ok = True
        for s, d, lo, hi in ((sx, dx, xmin, xmax), (sy, dy, ymin, ymax)):
            if abs(d) < 1e-9:
                if s < lo or s > hi:
                    ok = False
                    break
                continue
            t1 = (lo - s) / d
            t2 = (hi - s) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                ok = False
                break
        if ok and tmin > 1e-4 and tmin <= 1.0:
            best_t = tmin if best_t is None else min(best_t, tmin)
    return best_t * seg_len if best_t is not None else math.inf


def geodesic_bfs(start, goal, walls):
    xmin, xmax, ymin, ymax = BOUNDS
    nx = int(round((xmax - xmin) / CELL))
    ny = int(round((ymax - ymin) / CELL))
    # bytearray of size nx*ny (row-major: idx = x*ny + y)
    blocked = bytearray(nx * ny)
    for px, py, hx, hy in walls:
        ix_lo = max(0, int((px - hx - CAR_RADIUS_PAD - xmin) / CELL))
        ix_hi = min(nx, int((px + hx + CAR_RADIUS_PAD - xmin) / CELL) + 1)
        iy_lo = max(0, int((py - hy - CAR_RADIUS_PAD - ymin) / CELL))
        iy_hi = min(ny, int((py + hy + CAR_RADIUS_PAD - ymin) / CELL) + 1)
        for ix in range(ix_lo, ix_hi):
            base = ix * ny
            for iy in range(iy_lo, iy_hi):
                blocked[base + iy] = 1

    def cell(p):
        return (
            min(max(int((p[0] - xmin) / CELL), 0), nx - 1),
            min(max(int((p[1] - ymin) / CELL), 0), ny - 1),
        )

    sg, gg = cell(start), cell(goal)
    if blocked[sg[0] * ny + sg[1]]:
        blocked[sg[0] * ny + sg[1]] = 0
    if blocked[gg[0] * ny + gg[1]]:
        blocked[gg[0] * ny + gg[1]] = 0

    dist = [-1] * (nx * ny)
    dist[gg[0] * ny + gg[1]] = 0
    q = deque([gg])
    while q:
        x, y = q.popleft()
        d = dist[x * ny + y]
        for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxn, nyn = x + ddx, y + ddy
            if 0 <= nxn < nx and 0 <= nyn < ny:
                idx = nxn * ny + nyn
                if not blocked[idx] and dist[idx] == -1:
                    dist[idx] = d + 1
                    q.append((nxn, nyn))
    sd = dist[sg[0] * ny + sg[1]]
    return math.inf if sd < 0 else sd * CELL


def main():
    rows = []
    for hop in ("hop_1", "hop_2", "hop_3"):
        for xml in sorted((SCENE_ROOT / hop / "benchmark_3").rglob("*.xml")):
            car, goal, walls = parse_scene(xml)
            if car is None or goal is None:
                continue
            E = math.hypot(goal[0] - car[0], goal[1] - car[1])
            hit = first_wall_hit(car, goal, walls)
            G = geodesic_bfs(car, goal, walls)
            ratio = G / E if E > 1e-6 else math.inf
            rel = str(xml).split("benchmark_3/")[1]
            rows.append((hop, rel, E, hit, G, ratio))

    # Sort by hop then by hit distance (closest = most suspicious first)
    rows.sort(key=lambda r: (r[0], r[3]))

    print(f"{'hop':<6} {'scene':<36} {'E':>6} {'hit':>7} {'G':>6} {'G/E':>5}")
    print("-" * 70)
    for hop, scene, E, hit, G, ratio in rows:
        hit_s = f"{hit:7.3f}" if math.isfinite(hit) else "    inf"
        ratio_s = f"{ratio:5.2f}" if math.isfinite(ratio) else "  inf"
        print(f"{hop:<6} {scene:<36} {E:6.3f} {hit_s} {G:6.3f} {ratio_s}")

    print("\n=== summary per hop ===")
    for hop in ("hop_1", "hop_2", "hop_3"):
        hop_rows = [r for r in rows if r[0] == hop]
        n = len(hop_rows)
        hit_close = sum(1 for r in hop_rows if r[3] <= 0.20)
        hit_any = sum(1 for r in hop_rows if math.isfinite(r[3]))
        high_ratio = sum(1 for r in hop_rows if math.isfinite(r[5]) and r[5] > 1.5)
        e_vals = [r[2] for r in hop_rows]
        print(
            f"  {hop}: N={n}  E={min(e_vals):.2f}-{max(e_vals):.2f}m  "
            f"line-hits-wall={hit_any}  hit<=0.20m={hit_close}  G/E>1.5={high_ratio}"
        )


if __name__ == "__main__":
    main()
