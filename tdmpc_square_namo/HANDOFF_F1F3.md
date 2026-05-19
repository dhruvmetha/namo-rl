# NAMO-RL F1+F3 reward fix — handoff for a second-server relaunch

You are an agent picking this up cold. The goal is to relaunch three TD-MPC²-Square training runs on a fresh GPU with a small reward-scheme fix that the in-flight runs have empirically demonstrated is needed. Everything you need to do is in this doc.

## 1. TL;DR — what to change and why

Three vanilla TD-MPC²-Square runs are training on `tns/tdmpc-square-namo` (wandb runs `4kammhz0` / `cwl1web3` / `h94ba4et`). They use a **bounded "tolerance"-style reward** (DMC point_mass-aligned: `r = near_target × small_ctrl ∈ [0, 1]/step`) with `goal_position_tol = 0.15 m`. After ~380k env steps (38% of the 1M budget), the empirical picture is:

| Suite | Cum train success | Eval `succ` history (out of 5) | Diagnosis |
| --- | ---: | --- | --- |
| default (`scenes/car/nav_env*`) | ~41% | 0.6 / 0 / 0.6 / 0.4 / **1.0** / 0.2 / 0.2 / 0.4 / 0.4 / 0.2 / 0.6 / 0.6 / **1.0** / 0.4 / 0.4 / 0.6 / 0.6 | works but oscillates wildly between 0 and 1.0 — "lingering trap" |
| `hop_1/benchmark_3` | ~17% | 0 / 0.2 / 0.2 / 0.2 / 0.6 / 0 / 0.6 / 0.4 / 0.6 / 0.6 / 0.2 / 0 | also oscillates, lower ceiling |
| `hop_2/benchmark_3` | **0% (0 / 5036 episodes ever)** | 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0 | completely stuck, wall-hit storm |

**Lingering trap (default and hop_1).** Bounded `near_target ≈ 1/step` plus `γ ≈ 0.99` means the discounted "hover just outside the goal-tolerance ball forever" return is ≈ `1/(1−γ) ≈ 100`. Crossing into the ball terminates the episode → future Q = 0. The deterministic planner correctly prefers the unbounded hover to the finite cross-and-terminate trajectory; training rollouts cross only because Gaussian exploration noise stumbles into the ball.

**Wall-hit storm (hop_2).** `reward_tolerance_wall_penalty = 0.0` (matching DMC point_mass) gives the agent zero explicit signal that walls are bad. In hop_2's tighter corridors the agent never disentangles "wall hit" from "regular step." Episodes average ~70 steps — agent dies fast, never reaches goal.

**The fix** — two reward-config overrides + one budget bump:

| Knob | Old value | New value | Why |
| --- | --- | --- | --- |
| `namo.reward_tolerance_terminal_bonus` | `0.0` | **`100.0`** | Beats the discounted-hover Q ≈ 99 so terminating becomes strictly preferred. `symlog(100) ≈ 4.6`, well inside `vmax = 10`, so no two-hot bin clipping. |
| `namo.reward_tolerance_wall_penalty`   | `0.0` | **`10.0`** | Explicit "walls are bad" signal — required for hop_2 to ever learn obstacle avoidance. |
| `namo.max_episode_steps` (hop_* only)  | `500` | **`750`** | Empirically ~32% of hop_1 episodes timed out at 500 steps; max successful length was 433. 750 lifts the success ceiling. **Default suite stays at 500** (only 5.2% truncated; max succ length 429). |

Keep everything else identical to the current runs: `reward_style=tolerance`, `goal_position_tol=0.15`, `seed=1`, `steps=1000000`, `model_size=5`, `eval_freq=25000`, `eval_episodes=5`, `save_video=true`, `save_agent=true`, `eval_oracle_mpc=false`, `disable_wandb=false`.

## 1b. Design constraint — train one world model per hop level

**Never mix hop levels (hop_1 / hop_2 / hop_3) inside a single training run.** Each hop level is a distinct difficulty/geometry distribution and should get its own dedicated world model + Q/π. The structure we want is:

- **3 separate runs** = 3 separate world models, each trained only on its own hop level's scene pool.
- **Within a single hop level**, sampling across `run_NNNN/env_*_pair_MMM.xml` (multiple worlds × multiple start-goal tasks) is fine and intended — that's `namo.recursive_scene_glob=true` pointed at `hop_X/benchmark_3/`.
- **Across hop levels** is not. Do not point `scene_dir` at `external/namo-rl/scenes/car/` and recursively glob all of `hop_*` together. Do not create a multitask config that interleaves them. Do not warm-start a hop_2 run from a hop_1 checkpoint.

Concretely: each of the three launch commands in §6 below targets exactly one hop level (or the default suite) — keep it that way. If you ever add hop_3, give it its own fourth run.

The rationale is that hop levels have meaningfully different obstacle topology and start-goal distances, and a shared world model would average across distributions that the agent needs to treat separately. The empirical hop_1-vs-hop_2 gap (16.5% vs 0% cum success at the same step count under identical hyperparameters) is direct evidence that these are different tasks, not different instances of one task.

## 2. Code layout — what already exists

The repo lives at `/home/kowndi/Documents/spring26/td-mpc-extensions/` on the current machine. The key folder is `tdmpc_square_namo/`:

```
tdmpc_square_namo/
  setup.py
  README.md
  tdmpc_square_namo/
    __init__.py
    config.yaml          # Hydra config — F1+F3 knobs already wired here
    train.py             # entrypoint
    video.py             # LocalVideoRecorder (wandb-free mp4 + wandb dual-sink)
    envs/
      __init__.py
      namo.py            # adapter — already supports reward_tolerance_{terminal_bonus,wall_penalty}
  tests/
    smoke_env.py
    smoke_render.py
  screen_logs/           # logs from the previous runs (keep for reference)
```

**Critical**: `reward_style="tolerance"`, `reward_tolerance_terminal_bonus`, `reward_tolerance_wall_penalty`, and `recursive_scene_glob` are *already implemented* in `tdmpc_square_namo/envs/namo.py` and exposed in `config.yaml` under the `namo:` block. You only need to override them on the CLI — no code changes required for the fix itself.

The upstream `tdmpc_square` package (agent, buffer, trainer, planner) is at `../tdmpc_square/` and is imported via `sys.path` fixups inside `train.py` — no `pip install` needed.

NAMO-RL scenes are at `../external/namo-rl/scenes/car/`. Each `hop_X/benchmark_3/run_NNNN/env_*_pair_MMM.xml` is a (start, goal) task on a fixed obstacle layout. See `external/namo-rl/README.md` for details.

## 3. Getting the code onto the second server

Two options:

**Option A — sync from this machine.** From the second server (assuming SSH access to this one):

```bash
mkdir -p ~/Documents/spring26 && cd ~/Documents/spring26
rsync -av --exclude '__pycache__' --exclude 'logs' --exclude 'wandb' \
      --exclude 'screen_logs' --exclude '*.egg-info' \
      kowndi@<this-host>:/home/kowndi/Documents/spring26/td-mpc-extensions/ \
      td-mpc-extensions/
```

You need at minimum:
- `tdmpc_square/` (the upstream package, no code changes from us)
- `tdmpc_square_namo/` (our adapter + entrypoint)
- `external/namo-rl/` (the env + scenes — the hop suites are nested XMLs)

**Option B — git clone.** The repo is the td-mpc-extensions worktree. If you have a remote, clone it; otherwise option A is simpler. `external/namo-rl/` may need to be cloned separately from its own repo.

## 4. Environment setup on the second server

Conda env requirements (the current machine has a `tdmpc-square` env that works):

```bash
conda create -n tdmpc-square python=3.11 -y
conda activate tdmpc-square
pip install torch==2.3.1 mujoco==3.1.6 gymnasium==0.29.1 hydra-core==1.3.2 \
            wandb==0.26.0 termcolor tensordict dm_control imageio imageio-ffmpeg
```

(Versions confirmed working on the current machine. Other compatible combos may also work.)

For wandb auth on the second server, the user's account is `kb1204` (`tns` org). Run `wandb login` once with the user's API key, or rsync `~/.netrc` from the current machine.

## 5. Smoke tests before kicking off training

Run both from the `tdmpc_square_namo/` folder:

```bash
cd tdmpc_square_namo
MUJOCO_GL=egl python tests/smoke_env.py
# Expected: obs_space Box((76,)), act_space Box((2,)), 50 random steps without errors,
#           "SMOKE OK"

MUJOCO_GL=egl python tests/smoke_render.py
# Expected: 320×240×3 frame, writes a small mp4, "RENDER SMOKE OK"
```

If both pass, the env is wired correctly. If `smoke_env.py` fails with `MuJoCo` errors, you likely need `MUJOCO_GL=egl` plus an EGL-capable GPU driver.

## 6. Launch the three F1+F3 runs

From `tdmpc_square_namo/` (i.e. the outer folder that contains the inner `tdmpc_square_namo/` package + `setup.py`):

```bash
# Repo paths
ROOT=$(pwd)/../external/namo-rl/scenes/car
PYBIN=$(which python)
mkdir -p screen_logs

# 1) default — keep max_episode_steps=500 (no override needed)
screen -dmS namo_train_f1f3 bash -c "
$PYBIN -u -m tdmpc_square_namo.train \
  task=namo_car seed=1 steps=1000000 eval_freq=25000 eval_episodes=5 \
  save_video=true save_agent=true eval_oracle_mpc=false model_size=5 \
  device='cuda:0' \
  exp_name='vanilla_namo_default_seed1_tol_gtol15_tb100_wp10' \
  namo.reward_style=tolerance namo.goal_position_tol=0.15 \
  namo.reward_tolerance_terminal_bonus=100.0 \
  namo.reward_tolerance_wall_penalty=10.0 \
  2>&1 | tee screen_logs/f1f3_default_seed1.log
  echo '--- run ended; Ctrl+D to leave ---'
  exec bash
"

# 2) hop_1 — bump max_episode_steps to 750
screen -dmS namo_hop1_f1f3 bash -c "
$PYBIN -u -m tdmpc_square_namo.train \
  task=namo_car seed=1 steps=1000000 eval_freq=25000 eval_episodes=5 \
  save_video=true save_agent=true eval_oracle_mpc=false model_size=5 \
  device='cuda:0' \
  exp_name='vanilla_namo_hop1_seed1_tol_gtol15_tb100_wp10_eps750' \
  namo.reward_style=tolerance namo.goal_position_tol=0.15 \
  namo.reward_tolerance_terminal_bonus=100.0 \
  namo.reward_tolerance_wall_penalty=10.0 \
  namo.max_episode_steps=750 \
  namo.scene_dir=$ROOT/hop_1/benchmark_3 namo.recursive_scene_glob=true \
  2>&1 | tee screen_logs/f1f3_hop1_seed1.log
  echo '--- run ended; Ctrl+D to leave ---'
  exec bash
"

# 3) hop_2 — same as hop_1, just different scene dir
screen -dmS namo_hop2_f1f3 bash -c "
$PYBIN -u -m tdmpc_square_namo.train \
  task=namo_car seed=1 steps=1000000 eval_freq=25000 eval_episodes=5 \
  save_video=true save_agent=true eval_oracle_mpc=false model_size=5 \
  device='cuda:0' \
  exp_name='vanilla_namo_hop2_seed1_tol_gtol15_tb100_wp10_eps750' \
  namo.reward_style=tolerance namo.goal_position_tol=0.15 \
  namo.reward_tolerance_terminal_bonus=100.0 \
  namo.reward_tolerance_wall_penalty=10.0 \
  namo.max_episode_steps=750 \
  namo.scene_dir=$ROOT/hop_2/benchmark_3 namo.recursive_scene_glob=true \
  2>&1 | tee screen_logs/f1f3_hop2_seed1.log
  echo '--- run ended; Ctrl+D to leave ---'
  exec bash
"

screen -ls
```

Confirm each screen reaches its `Learnable parameters: 4,955,490` line and the `Video recorder active (local mp4 + wandb)` line within ~30 s. The wandb run URLs print near the top of each log (under `wandb: 🚀 View run at https://wandb.ai/tns/tdmpc-square-namo/runs/<id>`).

If `cuda:0` is not the right device on the second server, change all three `device='cuda:0'` overrides.

## 7. What to expect once running

These are the empirical predictions; revise them if you observe different behaviour.

- **default**: should converge much faster to high eval success (4-5 / 5) without the 0/1 oscillation. The +100 terminal bonus should make termination strictly preferred over hovering, so the value function won't flip-flop.
- **hop_1**: should also stabilise. Some oscillation may remain since the env still has structural diversity (28 XMLs / 5 worlds), but eval succ should hover ≥ 2-3 / 5 once warmed up. The 750-step budget removes the ~32% truncation tax.
- **hop_2**: this is the most interesting test. The wall penalty should snap the agent out of the wall-hit storm. Expect ep_len to climb from ~70 toward 200+ and the first eval successes to land somewhere between 100k-300k steps. **If hop_2 is still 0/5 at 250k steps, the diagnosis was incomplete** — escalate to the user.

Anomaly thresholds to flag (same as the live-run monitor): `state != "running"`, training success rate dropping >50% vs prior bucket, `eval/episode_reward` negative, `episode_length < 50`, NaN/Traceback in screen logs, process death.

## 8. Useful references in this repo

These files in the project memory already capture related context — read them if you want more background:

- `~/.claude/projects/-home-kowndi-Documents-spring26-td-mpc-extensions/memory/feedback_namo_runs.md` — launch protocol (wandb + screen).
- `~/.claude/projects/.../project_namo_active_runs.md` — wandb run IDs of the runs being replaced.
- `~/.claude/projects/.../project_namo_hop_budget.md` — empirical 32% truncation finding for hop_1.
- `~/.claude/projects/.../project_namo_tolerance_lingering_pending.md` — the lingering-trap analysis these fixes address.
- `~/.claude/projects/.../project_namo_env_design.md` — train/test env design decisions.
- `~/.claude/projects/.../project_namo_reward_ablation_pending.md` — the two reward-style variants (additive vs tolerance).

These memory files live on the *current* machine. On the second server they won't be present, but this handoff doc is self-contained.

## 9. What to do when the runs finish (or hit 1M steps)

1. Verify the three runs hit the expected eval success targets above.
2. Update `project_namo_active_runs.md` on the original machine with the new wandb run IDs.
3. Pull the eval-success curves and final 5-eval-episode success rates for the three suites — that's the headline data the user will want.
4. Compare against the killed tolerance-only runs (4kammhz0 / cwl1web3 / h94ba4et) to quantify the F1+F3 fix's contribution.

Good luck.
