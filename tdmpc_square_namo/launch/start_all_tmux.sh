#!/bin/bash
# Launch three F1+F3 namo training tmux sessions (one per GPU).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

# Kill any pre-existing sessions with our names so this is idempotent.
for S in namo_f1f3_hop1 namo_f1f3_hop2 namo_f1f3_hop3; do
    if tmux has-session -t "$S" 2>/dev/null; then
        echo "killing existing tmux session $S"
        tmux kill-session -t "$S"
    fi
done

# Create three detached tmux sessions, each running its hop-category script.
# Trailing 'exec bash' keeps the pane open if the python process exits.
tmux new-session -d -s namo_f1f3_hop1
tmux send-keys -t namo_f1f3_hop1 "bash $HERE/f1f3_hop1.sh; echo '--- hop1 ended; Ctrl+D to leave ---'; exec bash" Enter

tmux new-session -d -s namo_f1f3_hop2
tmux send-keys -t namo_f1f3_hop2 "bash $HERE/f1f3_hop2.sh; echo '--- hop2 ended; Ctrl+D to leave ---'; exec bash" Enter

tmux new-session -d -s namo_f1f3_hop3
tmux send-keys -t namo_f1f3_hop3 "bash $HERE/f1f3_hop3.sh; echo '--- hop3 ended; Ctrl+D to leave ---'; exec bash" Enter

echo "--- tmux sessions ---"
tmux ls
echo "--- logs at $HERE/../screen_logs/ ---"
