#!/usr/bin/env bash
# 决定性重跑: B1-B5 x 12 种子 (40-51), 已修 set_seed -> 训练确定性, 同种子各机制
# 共享网络初始化 -> 机制效应干净隔离. 单线程(OMP=1)保证浮点确定, 16 并行占满核.
#   cd /home/ylj/CPDRE && bash cleanmarl/_run_exp2_seeded.sh
set -uo pipefail
cd /home/ylj/CPDRE
export PY=/home/ylj/miniconda3/envs/marllib_4060/bin/python
export CONFIG=configs/happo_cpdre.yaml
export DEVICE=cpu
JOBS=16
SEEDS="40 41 42 43 44 45 46 47 48 49 50 51"
mkdir -p logs/sweeplogs
echo "[exp2-seeded] B1-B5 | 12种子 | ${JOBS}进程 x 1线程(确定性) | $DEVICE"
START=$(date +%s)
for g in B1 B2 B3 B4 B5; do for s in $SEEDS; do echo "$g $s"; done; done \
 | xargs -P "$JOBS" -n 2 bash -c '
     g="$0"; s="$1"
     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       "$PY" cleanmarl/run_sweep.py --config "$CONFIG" --group "$g" --seeds "$s" --device "$DEVICE" \
       > "logs/sweeplogs/${g}_${s}.log" 2>&1 \
       && echo "  [ok] $g seed$s" || echo "  [FAIL] $g seed$s"
   '
echo "[exp2-seeded] 完成, 用时 $(( $(date +%s) - START )) 秒"
