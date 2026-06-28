#!/usr/bin/env bash
# 非稳态季节实验: 旺季起始/时长/幅度逐年随机(决策时不可观测), 合同/规则按名义日历预承诺.
# B1-B5 x 12种子. 看动态策略(B2/B4)是否因适应实现压力而拉开静态长协(B5)/规则(B1).
set -uo pipefail
cd /home/ylj/CPDRE
export PY=/home/ylj/miniconda3/envs/marllib_4060/bin/python
export CONFIG=configs/happo_cpdre_nonstat.yaml
export DEVICE=cpu
JOBS=16
SEEDS="40 41 42 43 44 45 46 47 48 49 50 51"
mkdir -p logs/sweeplogs
echo "[nonstat] B1-B5 | 12种子 | 16进程x1线程(确定性)"
START=$(date +%s)
for g in B1 B2 B3 B4 B5; do for s in $SEEDS; do echo "$g $s"; done; done \
 | xargs -P "$JOBS" -n 2 bash -c '
     g="$0"; s="$1"
     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       "$PY" cleanmarl/run_sweep.py --config "$CONFIG" --group "$g" --seeds "$s" --device "$DEVICE" \
       > "logs/sweeplogs/NS_${g}_${s}.log" 2>&1 \
       && echo "  [ok] $g seed$s" || echo "  [FAIL] $g seed$s"
   '
echo "[nonstat] 完成, 用时 $(( $(date +%s) - START )) 秒"
