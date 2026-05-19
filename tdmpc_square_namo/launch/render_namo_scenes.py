#!/usr/bin/env python
"""Render annotated top-down SVG schematics of every hop_N namo scene.

Output:
  tdmpc_square_namo/scene_audit/{hop_N}/<scene_basename>.svg  -- one per XML
  tdmpc_square_namo/scene_audit/index.html                    -- gallery sorted
                                                                  by "shortcut
                                                                  badness"
                                                                  (small hit
                                                                  first)

Pure stdlib (works on amarel1 with /usr/bin/python3.6 -- no numpy/matplotlib).
"""

import math
import os
import subprocess
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

# Reuse the same geometry routines as the audit script via import-by-path.
import importlib.util
_audit_spec = importlib.util.spec_from_file_location(
    "audit_namo_scenes",
    str(Path(__file__).parent / "audit_namo_scenes.py"),
)
_audit = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(_audit)
parse_scene = _audit.parse_scene
first_wall_hit = _audit.first_wall_hit
geodesic_bfs = _audit.geodesic_bfs

SCENE_ROOT = Path(
    "/cache/home/kb1204/code/tdmpc_square_public/external/namo-rl/scenes/car"
)
OUT_ROOT = Path(
    "/cache/home/kb1204/code/tdmpc_square_public/tdmpc_square_namo/scene_audit"
)
GOAL_TOL = 0.05  # current namo.goal_position_tol

# SVG layout
VIEW_HALF = 0.75       # world half-extent shown (m)
PX = 520               # svg pixel size
MARGIN = 18            # px reserved at top for title bar
PLOT_PX = PX - MARGIN * 2


def w2s(p):
    """World (x, y) -> svg (px_x, px_y)."""
    wx, wy = p
    sx = (wx + VIEW_HALF) / (2 * VIEW_HALF) * PLOT_PX + MARGIN
    sy = (VIEW_HALF - wy) / (2 * VIEW_HALF) * PLOT_PX + MARGIN
    return sx, sy


def w2s_len(d):
    return d / (2 * VIEW_HALF) * PLOT_PX


def parse_obstacles(xml_path):
    """Return list of (px, py, hx, hy, euler_deg) for obstacle_*_movable geoms."""
    root = ET.parse(xml_path).getroot()
    obs = []
    for geom in root.iter("geom"):
        n = geom.get("name") or ""
        if "_movable" in n:
            pos = list(map(float, geom.get("pos").split()))
            size = list(map(float, geom.get("size").split()))
            euler_attr = geom.get("euler", "0 0 0").split()
            yaw = float(euler_attr[2]) if len(euler_attr) >= 3 else 0.0
            obs.append((pos[0], pos[1], size[0], size[1], yaw))
    return obs


def parse_car_heading(xml_path):
    """Best-effort car heading from <body name='car' quat='w x y z'>; default 0."""
    root = ET.parse(xml_path).getroot()
    for body in root.iter("body"):
        if body.get("name") == "car":
            quat_attr = body.get("quat")
            if quat_attr:
                w, x, y, z = map(float, quat_attr.split())
                # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
                yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                return yaw
            return 0.0
    return 0.0


def render_scene(xml_path, hop_dir):
    car, goal, walls = parse_scene(xml_path)
    if car is None or goal is None:
        return None
    movables = parse_obstacles(xml_path)
    yaw = parse_car_heading(xml_path)

    E = math.hypot(goal[0] - car[0], goal[1] - car[1])
    hit = first_wall_hit(car, goal, walls)
    G = geodesic_bfs(car, goal, walls)
    ratio = G / E if E > 1e-6 else math.inf

    badness = hit if math.isfinite(hit) else 99.0  # used for sorting

    # SVG elements
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PX}" height="{PX}" '
        f'viewBox="0 0 {PX} {PX}">',
        f'<rect x="0" y="0" width="{PX}" height="{PX}" fill="#fafafa"/>',
    ]

    # Arena boundary (the playable interior; outer walls drawn below will frame it)
    ax0, ay0 = w2s((-VIEW_HALF + 0.05, VIEW_HALF - 0.05))
    aw = w2s_len(2 * VIEW_HALF - 0.10)
    parts.append(
        f'<rect x="{ax0:.1f}" y="{ay0:.1f}" width="{aw:.1f}" height="{aw:.1f}" '
        f'fill="white" stroke="#dddddd" stroke-width="1"/>'
    )

    # Static walls (gray)
    for (px, py, hx, hy) in walls:
        x0, y0 = w2s((px - hx, py + hy))
        w_px = w2s_len(2 * hx)
        h_px = w2s_len(2 * hy)
        parts.append(
            f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w_px:.1f}" '
            f'height="{h_px:.1f}" fill="#888888" stroke="#444444" stroke-width="0.7"/>'
        )

    # Movables (yellow, rotated)
    for (px, py, hx, hy, yaw_deg) in movables:
        x0, y0 = w2s((px - hx, py + hy))
        w_px = w2s_len(2 * hx)
        h_px = w2s_len(2 * hy)
        cx, cy = w2s((px, py))
        parts.append(
            f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w_px:.1f}" '
            f'height="{h_px:.1f}" fill="#ffd95a" stroke="#aa7c10" '
            f'stroke-width="0.7" transform="rotate({-yaw_deg:.2f} {cx:.1f} {cy:.1f})"/>'
        )

    # Straight line start -> goal (dashed orange)
    cx, cy = w2s(car)
    gx, gy = w2s(goal)
    parts.append(
        f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{gx:.1f}" y2="{gy:.1f}" '
        f'stroke="#e8881a" stroke-width="1" stroke-dasharray="4,3"/>'
    )

    # If line hits a wall, mark hit point with a red X
    if math.isfinite(hit) and E > 1e-6:
        t = hit / E
        hx_w = car[0] + t * (goal[0] - car[0])
        hy_w = car[1] + t * (goal[1] - car[1])
        hxp, hyp = w2s((hx_w, hy_w))
        s = 6
        parts.append(
            f'<line x1="{hxp - s}" y1="{hyp - s}" x2="{hxp + s}" '
            f'y2="{hyp + s}" stroke="red" stroke-width="2"/>'
        )
        parts.append(
            f'<line x1="{hxp - s}" y1="{hyp + s}" x2="{hxp + s}" '
            f'y2="{hyp - s}" stroke="red" stroke-width="2"/>'
        )

    # Goal: tolerance ring + red dot
    tol_r = w2s_len(GOAL_TOL)
    parts.append(
        f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="{tol_r:.1f}" '
        f'fill="rgba(255,0,0,0.18)" stroke="#cc0000" stroke-width="0.8"/>'
    )
    parts.append(
        f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="3" fill="#cc0000"/>'
    )

    # Car: blue dot + heading arrow
    arrow_len = w2s_len(0.07)
    ax = cx + arrow_len * math.cos(yaw)
    ay = cy - arrow_len * math.sin(yaw)  # svg y flipped
    parts.append(
        f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{ax:.1f}" y2="{ay:.1f}" '
        f'stroke="#0050d0" stroke-width="2.4"/>'
    )
    parts.append(
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="#0050d0"/>'
    )

    # Title bar with metrics
    hit_s = f"{hit:.3f}" if math.isfinite(hit) else "inf"
    ratio_s = f"{ratio:.2f}" if math.isfinite(ratio) else "inf"
    title = (
        f"{hop_dir} | {xml_path.parent.name}/{xml_path.name}  "
        f"E={E:.3f}  hit={hit_s}  G={G:.3f}  G/E={ratio_s}"
    )
    parts.append(
        f'<text x="{MARGIN}" y="{MARGIN - 4}" font-family="monospace" '
        f'font-size="10" fill="#222">{title}</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts), {
        "hop": hop_dir,
        "rel": f"{xml_path.parent.name}/{xml_path.name}",
        "E": E,
        "hit": hit,
        "G": G,
        "ratio": ratio,
        "badness": badness,
    }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for hop in ("hop_1", "hop_2", "hop_3"):
        (OUT_ROOT / hop).mkdir(parents=True, exist_ok=True)
        for xml in sorted((SCENE_ROOT / hop / "benchmark_3").rglob("*.xml")):
            result = render_scene(xml, hop)
            if result is None:
                continue
            svg, meta = result
            stem = f"{xml.parent.name}__{xml.stem}"
            svg_path = OUT_ROOT / hop / f"{stem}.svg"
            png_path = OUT_ROOT / hop / f"{stem}.png"
            svg_path.write_text(svg)
            # Rasterize to PNG via rsvg-convert (pre-installed on amarel1).
            subprocess.run(
                ["rsvg-convert", "-w", "800", "-h", "800",
                 str(svg_path), "-o", str(png_path)],
                check=True,
            )
            meta["img_rel"] = f"{hop}/{stem}.png"
            rows.append(meta)

    # Sort by badness ascending (smallest hit = worst shortcut, shows first)
    rows.sort(key=lambda r: (r["badness"], r["E"]))

    # Build index.html
    html = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>namo scene audit gallery</title>",
        "<style>",
        "body{font-family:sans-serif;margin:14px;background:#222;color:#eee}",
        "h1{font-size:16px;margin:8px 0}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(540px,1fr));gap:14px}",
        ".cell{background:#333;padding:8px;border-radius:6px}",
        ".cell.bad{background:#5a2222}",
        ".cell.warn{background:#5a4a22}",
        ".meta{font-family:monospace;font-size:12px;color:#ddd;margin-bottom:4px}",
        "img,svg{max-width:100%;display:block}",
        "</style></head><body>",
        "<h1>namo scene audit gallery &mdash; sorted by shortcut badness "
        "(smallest <code>hit</code> first)</h1>",
        "<p style='font-family:monospace;font-size:12px'>",
        "Legend: gray=static wall, yellow=movable, blue=car (line=heading), "
        "red dot=goal (light ring = goal_position_tol=0.05m), "
        "dashed orange=straight line to goal, red X=first wall hit on that line.</p>",
        "<div class='grid'>",
    ]
    for r in rows:
        cls = "cell"
        if math.isfinite(r["hit"]) and r["hit"] <= 0.20 and r["E"] <= 0.50:
            cls = "cell bad"
        elif math.isfinite(r["hit"]) and r["hit"] <= 0.20:
            cls = "cell warn"
        hit_s = f"{r['hit']:.3f}" if math.isfinite(r["hit"]) else "inf"
        ratio_s = f"{r['ratio']:.2f}" if math.isfinite(r["ratio"]) else "inf"
        html.append(f"<div class='{cls}'>")
        html.append(
            f"<div class='meta'>{r['hop']} | {r['rel']}<br>"
            f"E={r['E']:.3f}  hit={hit_s}  G={r['G']:.3f}  G/E={ratio_s}</div>"
        )
        html.append(f"<img src='{r['img_rel']}' alt='{r['rel']}'>")
        html.append("</div>")
    html.append("</div></body></html>")
    (OUT_ROOT / "index.html").write_text("\n".join(html))

    bad = sum(
        1 for r in rows
        if math.isfinite(r["hit"]) and r["hit"] <= 0.20 and r["E"] <= 0.50
    )
    warn = sum(
        1 for r in rows
        if math.isfinite(r["hit"]) and r["hit"] <= 0.20 and not (r["E"] <= 0.50)
    )
    print(
        f"wrote {len(rows)} SVGs + index.html to {OUT_ROOT}\n"
        f"  bad (E<=0.50 AND hit<=0.20): {bad}\n"
        f"  warn (hit<=0.20 only):       {warn}\n"
        f"  rest:                        {len(rows) - bad - warn}"
    )


if __name__ == "__main__":
    main()
