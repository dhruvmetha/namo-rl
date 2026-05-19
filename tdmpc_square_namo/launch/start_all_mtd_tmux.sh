#!/bin/bash
# Launch three mtd+randq+adaptive namo training tmux sessions (one per GPU).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

for S in namo_mtd_hop1 namo_mtd_hop2 namo_mtd_hop3; do
    if tmux has-session -t "$S" 2>/dev/null; then
        echo "killing existing tmux session $S"
        tmux kill-session -t "$S"
    fi
done

tmux new-session -d -s namo_mtd_hop1
tmux send-keys -t namo_mtd_hop1 "bash $HERE/mtd_hop1.sh; echo '--- mtd hop1 ended; Ctrl+D to leave ---'; exec bash" Enter

tmux new-session -d -s namo_mtd_hop2
tmux send-keys -t namo_mtd_hop2 "bash $HERE/mtd_hop2.sh; echo '--- mtd hop2 ended; Ctrl+D to leave ---'; exec bash" Enter

tmux new-session -d -s namo_mtd_hop3
tmux send-keys -t namo_mtd_hop3 "bash $HERE/mtd_hop3.sh; echo '--- mtd hop3 ended; Ctrl+D to leave ---'; exec bash" Enter

echo "--- tmux sessions ---"
tmux ls
echo "--- logs at $HERE/../screen_logs/mtd_*_seed1.log ---"
