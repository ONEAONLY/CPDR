#!/usr/bin/env bash
# 稀缺压力测试: K=3.0 (旺季需求3.6 > K -> 真稀缺), B2/B3/B4 x 12种子.
# 看互惠的系统级效应是否"条件性存在"(稀缺时才显现).
set -uo pipefail
cd /home/ylj/CPDRE
export PY=/home/ylj/miniconda3/envs/marllib_4060/bin/python
export CONFIG=configs/happo_cpdre_K30.yaml
export DEVICE=cpu
JOBS=16
SEEDS="40 41 42 43 44 45 46 47 48 49 50 51"
mkdir -p logs/sweeplogs
echo "[stress K=3.0] B2 B3 B4 | 12种子 | 16进程x1线程"
START=$(date +%s)
for g in B2 B3 B4; do for s in $SEEDS; do echo "$g $s"; done; done \
 | xargs -P "$JOBS" -n 2 bash -c '
     g="$0"; s="$1"
     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       "$PY" cleanmarl/run_sweep.py --config "$CONFIG" --group "$g" --seeds "$s" --device "$DEVICE" \
       > "logs/sweeplogs/K30_${g}_${s}.log" 2>&1 \
       && echo "  [ok] $g seed$s" || echo "  [FAIL] $g seed$s"
   '
echo "[stress] 完成, 用时 $(( $(date +%s) - START )) 秒"
