#!/usr/bin/env bash
# 自包含启动器: 只跑实验一的学习组 A3/A4/A5/A6 (A1/A2 规则基线已完成不重跑).
# 组别写死, 无需任何参数/环境变量 (规避后台启动层对带空格变量的转义问题).
#   cd /home/ylj/CPDRE && bash cleanmarl/_run_learning.sh
set -uo pipefail
cd /home/ylj/CPDRE
export PY=/home/ylj/miniconda3/envs/marllib_4060/bin/python
export CONFIG=configs/happo_cpdre.yaml
export THREADS=2
export DEVICE=cpu
JOBS=8
mkdir -p logs/sweeplogs
echo "[learning] 组别: A3 A4 A5 A6 | 种子 40-44 | ${JOBS}进程 x ${THREADS}线程 | $DEVICE"
START=$(date +%s)
for g in A3 A4 A5 A6; do for s in 40 41 42 43 44; do echo "$g $s"; done; done \
 | xargs -P "$JOBS" -n 2 bash -c '
     g="$0"; s="$1"
     OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
       "$PY" cleanmarl/run_sweep.py --config "$CONFIG" --group "$g" --seeds "$s" --device "$DEVICE" \
       > "logs/sweeplogs/${g}_${s}.log" 2>&1 \
       && echo "  [ok]   $g seed$s" \
       || echo "  [FAIL] $g seed$s -> logs/sweeplogs/${g}_${s}.log"
   '
echo "[learning] 全部完成, 用时 $(( $(date +%s) - START )) 秒"
