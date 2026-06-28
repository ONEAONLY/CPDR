#!/usr/bin/env bash
# 零参数启动器: 补跑 B5 固定优先/长协 5 种子 (HAPPO + ValueNorm, 与 B2-B4 同口径).
#   cd /home/ylj/CPDRE && bash cleanmarl/_run_b5.sh
set -uo pipefail
cd /home/ylj/CPDRE
export PY=/home/ylj/miniconda3/envs/marllib_4060/bin/python
export CONFIG=configs/happo_cpdre.yaml
export THREADS=2; export DEVICE=cpu
JOBS=5
mkdir -p logs/sweeplogs
echo "[B5] 固定优先/长协 | 种子 40-44 | ${JOBS}进程 x ${THREADS}线程 | $DEVICE"
START=$(date +%s)
for s in 40 41 42 43 44; do echo "B5 $s"; done \
 | xargs -P "$JOBS" -n 2 bash -c '
     g="$0"; s="$1"
     OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
       "$PY" cleanmarl/run_sweep.py --config "$CONFIG" --group "$g" --seeds "$s" --device "$DEVICE" \
       > "logs/sweeplogs/${g}_${s}.log" 2>&1 \
       && echo "  [ok] $g seed$s" || echo "  [FAIL] $g seed$s"
   '
echo "[B5] 完成, 用时 $(( $(date +%s) - START )) 秒"
