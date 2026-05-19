#!/bin/bash
OUT="${1:-/cache/home/kb1204/code/tdmpc_square_public/tdmpc_square_namo/launch/_panes.txt}"
{
echo "===SESSIONS==="
tmux list-sessions
for s in namo_f1f3_default namo_f1f3_hop1 namo_f1f3_hop2; do
    echo "===PANE $s==="
    tmux capture-pane -t "$s" -p -S -200 | grep -v "^$" | tail -30
done
echo "===CAPDONE==="
} > "$OUT" 2>&1
echo "wrote $OUT"
