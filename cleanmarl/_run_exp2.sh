#!/usr/bin/env bash
# 零参数启动器: 实验二 B1/B2/B3/B4 (B1 派生基线, B2/B3/B4 学习).
# 组别/参数全写死, 规避后台启动层对带参命令的转义问题.
#   cd /home/ylj/CPDRE && bash cleanmarl/_run_exp2.sh
set -uo pipefail
cd /home/ylj/CPDRE
export PY=/home/ylj/miniconda3/envs/marllib_4060/bin/python
export CONFIG=configs/happo_cpdre.yaml
export THREADS=2
export DEVICE=cpu
JOBS=8
mkdir -p logs/sweeplogs
echo "[exp2] 组别: B1 B2 B3 B4 | 种子 40-44 | ${JOBS}进程 x ${THREADS}线程 | $DEVICE"
START=$(date +%s)
for g in B1 B2 B3 B4; do for s in 40 41 42 43 44; do echo "$g $s"; done; done \
 | xargs -P "$JOBS" -n 2 bash -c '
     g="$0"; s="$1"
     OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
       "$PY" cleanmarl/run_sweep.py --config "$CONFIG" --group "$g" --seeds "$s" --device "$DEVICE" \
       > "logs/sweeplogs/${g}_${s}.log" 2>&1 \
       && echo "  [ok]   $g seed$s" \
       || echo "  [FAIL] $g seed$s -> logs/sweeplogs/${g}_${s}.log"
   '
echo "[exp2] 全部完成, 用时 $(( $(date +%s) - START )) 秒"
