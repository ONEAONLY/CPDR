#!/usr/bin/env bash
# GPU 版启动器: 学习组 A3/A4/A5/A6 在 RTX 4060 上跑.
# 单卡共享, 故并行数压到 4 (每进程 CUDA context ~0.6-1GB, 4 个稳妥 < 6GB 空闲).
#   cd /home/ylj/CPDRE && bash cleanmarl/_run_learning_gpu.sh
set -uo pipefail
cd /home/ylj/CPDRE
export PY=/home/ylj/miniconda3/envs/marllib_4060/bin/python
export CONFIG=configs/happo_cpdre.yaml
export THREADS=2
export DEVICE=cuda
JOBS=4
mkdir -p logs/sweeplogs
echo "[learning-gpu] 组别: A3 A4 A5 A6 | 种子 40-44 | ${JOBS}进程共享GPU x ${THREADS}线程 | $DEVICE"
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
echo "[learning-gpu] 全部完成, 用时 $(( $(date +%s) - START )) 秒"
