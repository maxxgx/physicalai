#!/bin/sh
# The MANIFEST's DEMO command, literally, for every declared type x bucket (r09 D4 / R-36: it raised for every pair).
# usage: demo_all_pairs.sh DELIVERY_OR_SCRIPTS_DIR KIT OUT_DIR   (seeds 9101-9112, default variant)
S="$1"; K="$2"; O="$3"; mkdir -p "$O"
seeds=""; pairs=""; i=9101
for t in resistor cpu battery; do for b in black green red blue; do seeds="$seeds,$i"; pairs="$pairs,$t:$b"; i=$((i+1)); done; done
cd /tmp && REAL2SIM_ROOT="$HOME/real2sim-data/bucket-v2" KIT="$K" "$HOME/physicalai-wt-bucket-v2/.venv/bin/python" -I -B "$S/bucketsim_v2.py" demo \
  --seeds "${seeds#,}" --pairs "${pairs#,}" --variant default --out "$O/demo_all_pairs.json" --dump "$O/dump" > "$O/demo_all_pairs.log" 2>&1
echo "exit $?"; tail -1 "$O/demo_all_pairs.log"
