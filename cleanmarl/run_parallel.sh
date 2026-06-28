#!/usr/bin/env bash
# 并行扫描运行器 (cleanmarl 训练器本身是单进程, 无 num_workers;
# 这里把每个 (组别, 种子) 当独立进程并行, 用满多核).
#
# 用法:
#   cd /home/ylj/CPDRE
#   bash cleanmarl/run_parallel.sh <exp> <config> <jobs> <threads> <device> <seeds>
# 例:
#   bash cleanmarl/run_parallel.sh exp1 configs/happo_cpdre.yaml 8 2 cpu
#   bash cleanmarl/run_parallel.sh exp2 configs/happo_cpdre.yaml 8 2 cpu 40,41,42,43,44
#
# 16 逻辑核建议: jobs*threads <= 16 (如 8 进程 x 2 线程, 或 6 x 2).
set -uo pipefail

EXP="${1:-exp1}"
CONFIG="${2:-configs/happo_cpdre.yaml}"
export JOBS="${3:-8}"          # 并行进程数
export THREADS="${4:-2}"       # 每进程 CPU 线程数 (避免抢核)
export DEVICE="${5:-cpu}"      # cpu 并行吞吐通常优于 6+进程抢一块 GPU
SEEDS="${6:-40,41,42,43,44}"
export PY="${PY:-/home/ylj/miniconda3/envs/marllib_4060/bin/python}"
export CONFIG DEVICE

if [ -n "${ONLY_GROUPS:-}" ]; then
  GROUPS="$ONLY_GROUPS"   # 覆盖: 只跑指定组, 如 ONLY_GROUPS="A3 A4 A5 A6"
else
  case "$EXP" in
    exp1) GROUPS="A1 A2 A3 A4 A5 A6";;
    exp2) GROUPS="B1 B2 B3 B4";;
    exp3) GROUPS="C1 C2 C3 C4 C5 C6 C7";;
    *) echo "未知实验: $EXP (可选 exp1/exp2/exp3)"; exit 1;;
  esac
fi

mkdir -p logs/sweeplogs
echo "[$EXP] 组别: $GROUPS"
echo "      种子: $SEEDS | 并行: ${JOBS}进程 x ${THREADS}线程 | 设备: $DEVICE"
echo "      单任务日志: logs/sweeplogs/<组>_<种子>.log  | 结果: logs/cleanmarl/<组>_<算法>_seed<种子>/"

START=$(date +%s)
for g in $GROUPS; do for s in ${SEEDS//,/ }; do echo "$g $s"; done; done \
 | xargs -P "$JOBS" -n 2 bash -c '
     g="$0"; s="$1"
     OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
       "$PY" cleanmarl/run_sweep.py --config "$CONFIG" --group "$g" --seeds "$s" --device "$DEVICE" \
       > "logs/sweeplogs/${g}_${s}.log" 2>&1 \
       && echo "  [ok]   $g seed$s" \
       || echo "  [FAIL] $g seed$s  -> logs/sweeplogs/${g}_${s}.log"
   '
echo "[$EXP] 全部完成, 用时 $(( $(date +%s) - START )) 秒"
