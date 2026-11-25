#!/bin/zsh
# The bucket_sort_sim_v2 recording (RECORDING_PLAN.md): about 300 clean successes through Studio's own pipeline.
# usage: record_300.sh DELIVERY_DIR [EPISODES] [PROJECT]   (Studio on :7871 and the simulation on :8931 already running)
set -e
D=${1:?delivery folder}; N=${2:-300}; P=${3:-bucket_sort_sim_v2}
LOG=~/real2sim-data/bucket-v2/sim/record_${P}_$(date +%Y%m%d_%H%M).log
mkdir -p /tmp/r2s_v2_run && cd /tmp/r2s_v2_run
~/physicalai-wt-bucket-v2/.venv/bin/python -I -B "$D/scripts/studio_record_v2.py" --episodes "$N" --project "$P" --dataset "$P" \
  --keep perfect --camera-fps 60 --studio http://127.0.0.1:7871 --sim http://127.0.0.1:8931 --log "$LOG"
echo "log: $LOG"
