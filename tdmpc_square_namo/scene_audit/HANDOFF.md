# NAMO scene audit — handoff

Context for the next session / a future you. All paths are on `amarel1` under the shared NFS so they're visible from any compute node too.

## What this is

A one-pass audit + visualization of the 60 namo_car scenes under
`external/namo-rl/scenes/car/hop_{1,2,3}/benchmark_3/`, focused on detecting
the **"agent initialized close to a wall with goal just on the other side"
shortcut pathology** that came up while watching 2-hop training video. In that
pathology the dense Euclidean reward gradient points *through* the static wall;
the agent can drive straight at the goal, accrue rising near-target reward, hit
the wall, terminate with a small penalty, and net positive — a local maximum at
the wall.

The audit doesn't fix that — it just measures which scenes are most exposed to
it so they can be curated out of the training set.

## Where the artifacts live

```
/cache/home/kb1204/code/tdmpc_square_public/tdmpc_square_namo/
├── launch/
│   ├── audit_namo_scenes.py        # the validator (pure stdlib, runs anywhere)
│   └── render_namo_scenes.py       # SVG + PNG renderer (uses rsvg-convert)
└── scene_audit/
    ├── HANDOFF.md                  # this file
    ├── index.html                  # gallery (PNG thumbnails, sorted worst-first)
    ├── hop_1/                      # 28 .svg + 28 .png
    ├── hop_2/                      # 10 .svg + 10 .png
    └── hop_3/                      # 22 .svg + 22 .png
```

Each SVG/PNG is a top-down schematic of one scene, annotated with the metrics
in the title bar:

- gray rectangles = static walls (`wall_*` geoms)
- yellow rectangles (rotated) = movable obstacles (`obstacle_*_movable`)
- blue dot + line = car start position + heading
- red dot + light-red ring = goal position + `goal_position_tol=0.05` radius
- dashed orange = straight line from start to goal
- red X = first wall hit on that straight line (if any)

## The validator: `launch/audit_namo_scenes.py`

Pure stdlib Python (no numpy/matplotlib — works on the amarel1 login node where
glibc 2.17 blocks the conda env's numpy 2.4.4). For each scene XML, computes:

| metric | meaning |
|---|---|
| `E` | Euclidean(car_start, goal) in meters |
| `hit` | distance along the straight start→goal segment to the first static wall AABB intersection (`inf` if no wall blocks the direct line) |
| `G` | BFS geodesic on a 1 cm grid with 4 cm wall padding (car-radius approximation). **Movables are treated as passable** so `G` reflects only the static-wall detour cost, not anything that requires pushing. |
| `G/E` | detour ratio |

To run:
```bash
cd /cache/home/kb1204/code/tdmpc_square_public/tdmpc_square_namo
/usr/bin/python3.6 launch/audit_namo_scenes.py
```

Outputs a sorted table (per-hop, by `hit` ascending) and a summary.

## The renderer: `launch/render_namo_scenes.py`

Same metrics, plus generates one SVG per scene and rasterizes to PNG via
`rsvg-convert` (pre-installed at `/usr/bin/rsvg-convert` on amarel1). Builds
`scene_audit/index.html` sorted worst-first (smallest `hit` first) with two
warning tints:

- **red tint** = `E <= 0.50` AND `hit <= 0.20` (close-walled-off — textbook bad)
- **amber tint** = `hit <= 0.20` only

To run:
```bash
cd /cache/home/kb1204/code/tdmpc_square_public/tdmpc_square_namo
/usr/bin/python3.6 launch/render_namo_scenes.py
```

To browse the gallery locally:
```bash
rsync -av amarel1:/cache/home/kb1204/code/tdmpc_square_public/tdmpc_square_namo/scene_audit/ ./scene_audit/
open scene_audit/index.html
```

## Audit findings (run 2026-05-18)

| hop | N scenes | E range (m) | line-hits-wall | hit ≤ 0.20m | G/E > 1.5 |
|---|---:|---|---:|---:|---:|
| hop_1 | 28 | 0.24 – 1.31 | 12 | 4 | 10 |
| hop_2 | 10 | 0.29 – 1.25 | **10/10** | 4 | **10/10** |
| hop_3 | 22 | 0.33 – 1.52 | **22/22** | 8 | 21 |

Take-away: ALL hop_2 and hop_3 scenes have a wall between start and goal — that's
**by design**, the "hop" structure. The pathology is the subset where the wall
is *right in front of the agent's start*, which is when the shortcut is short
enough to compete with the detour reward.

### Worst 5 (E < 0.50 AND hit < 0.20)

| hop | scene | E | hit | G/E |
|---|---|---:|---:|---:|
| hop_1 | `run_0001/env_0001_pair_001.xml` | 0.242 | 0.127 | 5.50 |
| hop_1 | `run_0004/env_0004_pair_001.xml` | 0.373 | 0.145 | 4.07 |
| hop_2 | `run_0004/env_0004_pair_001.xml` | 0.292 | 0.093 | 7.41 |
| hop_3 | `run_0014/env_0014_pair_002.xml` | 0.332 | 0.145 | 5.63 |
| hop_3 | `run_0011/env_0011_pair_000.xml` | 0.439 | 0.187 | 4.10 |

### Threshold-vs-keep-count (for curation)

| filter | hop_1 keep | hop_2 keep | hop_3 keep | total |
|---|---:|---:|---:|---:|
| no filter | 28 | 10 | 22 | 60 |
| hit > 0.15m | 26 | 8 | 19 | 53 |
| hit > 0.20m | 24 | 6 | 14 | 44 |
| hit > 0.30m | 20 | 4 | 9 | 33 |
| (E ≥ 0.50 OR hit ≥ 0.20) | 26 | 8 | 20 | 54 |
| (E ≥ 0.50 OR hit ≥ 0.30) | 23 | 6 | 17 | 46 |

The compound filter `(E ≥ 0.50 OR hit ≥ 0.20)` most directly matches the
stated concern: drop only scenes that are *both* close in Euclidean *and* have
a wall right in front. Keeps 54/60 with minimal disruption to hop_2 (already
thin at 10).

## Recommended curation flow (uncommitted as of handoff)

1. Pick a threshold (e.g., `E ≥ 0.50 OR hit ≥ 0.20`).
2. Symlink the surviving scenes into `external/namo-rl/scenes/car/hop_N/benchmark_3_clean/` (no XML copying — symlinks keep storage zero-cost and let the original `benchmark_3/` stay intact).
3. Update `namo.scene_dir` in all launch scripts (`tdmpc_square_namo/launch/{f1f3,mtd}_hop{1,2,3}.sh`) to point at `benchmark_3_clean`.
4. Apply uniformly to both vanilla TD-MPC²-Square and Adaptive+MTD+RQ runs to keep the head-to-head comparison fair.

The actual symlink set hasn't been built yet — once you've eyeballed the
gallery and picked a threshold, that's a 5-line shell loop.

## Why this came up (broader context)

The shortcut-and-die pathology is real but **scene curation is only half the
fix.** The other half is in
`~/.claude/projects/-cache-home-kb1204-code-tdmpc-square-public/memory/project_namo_reward_shaping_options.md`,
which lays out three reward / planner knobs:

1. **Crank `namo.reward_tolerance_wall_penalty` 10 → 100–200.** Cheapest fix. Currently the per-step Euclidean reward accrued during a 50-step approach (~25–35) outscores the wall_penalty (10), so the shortcut nets positive. ≥ ~3× terminal_bonus makes any wall hit unambiguously bad. One-line per launch script.
2. **Geodesic distance in `near_target`.** Evaluated and **ruled out** for NAMO: precomputing the distance field only stays accurate if obstacles don't move, but in NAMO they do. Per-step BFS adds compute *and* fights the research goal of learning push dynamics. See the memory note for the full reasoning.
3. **Longer planner horizon.** Bump `horizon: 3` → 5 or 6 in `tdmpc_square_namo/config.yaml`. Lets MPC see wall termination before committing. Linear compute cost increase per planner step.

Recommend pairing scene curation (this audit's output) with knob 1, optionally knob 3.

## Related artifacts in this codebase

- **Reward-shaping notes (the 3 knobs):**
  `~/.claude/projects/-cache-home-kb1204-code-tdmpc-square-public/memory/project_namo_reward_shaping_options.md`

- **Scene audit notes (this work):**
  `~/.claude/projects/-cache-home-kb1204-code-tdmpc-square-public/memory/project_namo_scene_audit.md`

- **Run history through the maintenance window:**
  All 14 namo wandb runs across `tns/tdmpc-square-namo` were synced to the
  cloud before the cluster went into maintenance. See
  `~/.claude/projects/.../memory/project_{ri_interactive,namo_mtd}_session_active.md`
  for the run IDs and per-hop bucketing.

## Caveats baked into the audit

- `G` (geodesic) treats only static `wall_*` geoms as obstacles. It does **not** account for movables — by design, since pushing-through is what the agent should learn. So `G` is a lower bound on the true minimum-effort path when the optimal solution requires pushing.
- The straight-line `hit` check ignores the car's own footprint (treats car as a point). A scene with `hit = 0.05` means a point-particle drives 5 cm before hitting; the actual car (wheelbase 7.5 cm) hits even sooner. So `hit` is a slight over-estimate of the available distance.
- `goal_position_tol=0.05` was the value in the runs at the time of the audit (tightened from 0.15 earlier the same day). The tolerance ring in the SVGs reflects this. If you change `goal_position_tol`, regenerate the renders so the ring matches.
