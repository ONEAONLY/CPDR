#!/usr/bin/env bash
# 零参数启动器: 实验三 C1-C7 (b4 机制消融, 均 HAPPO + ValueNorm).
#   C1=完整b4参照 C2=disable_g_u C3=fixed_lambda=1 C4=freeze_memory
#   C5=freeze_theta C6=去互惠观测 C7=η=0(去ημg奖励)
#   cd /home/ylj/CPDRE && bash cleanmarl/_run_exp3.sh
set -uo pipefail
cd /home/ylj/CPDRE
export PY=/home/ylj/miniconda3/envs/marllib_4060/bin/python
export CONFIG=configs/happo_cpdre.yaml
export THREADS=2; export DEVICE=cpu
JOBS=8
mkdir -p logs/sweeplogs
echo "[exp3] C1-C7 | 种子 40-44 | ${JOBS}进程 x ${THREADS}线程 | $DEVICE"
START=$(date +%s)
for g in C1 C2 C3 C4 C5 C6 C7; do for s in 40 41 42 43 44; do echo "$g $s"; done; done \
 | xargs -P "$JOBS" -n 2 bash -c '
     g="$0"; s="$1"
     OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
       "$PY" cleanmarl/run_sweep.py --config "$CONFIG" --group "$g" --seeds "$s" --device "$DEVICE" \
       > "logs/sweeplogs/${g}_${s}.log" 2>&1 \
       && echo "  [ok] $g seed$s" || echo "  [FAIL] $g seed$s"
   '
echo "[exp3] 完成, 用时 $(( $(date +%s) - START )) 秒"
