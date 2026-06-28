#!/usr/bin/env bash
# 混合启动器: 学习组 A3/A4/A5/A6 同时用 GPU + CPU, 稳定加速.
# 关键: WSL 下消费级 GPU 无 MPS, 多 CUDA 进程并发会偶发 "unspecified launch failure".
#        故 GPU 只跑【单进程】(不并发=不崩), CPU 跑 7 进程, 共 8 并发.
#        按吞吐配比: GPU 串行慢 -> 只分 3 个作业; CPU -> 其余 17 个 (7 并行约 3 波).
#   cd /home/ylj/CPDRE && bash cleanmarl/_run_learning_hybrid.sh
set -uo pipefail
cd /home/ylj/CPDRE
export PY=/home/ylj/miniconda3/envs/marllib_4060/bin/python
export CONFIG=configs/happo_cpdre.yaml
export THREADS=2
mkdir -p logs/sweeplogs

# 20 个作业 (索引 0..19); 抽 3 个跨组的给 GPU(单进程), 其余给 CPU
all=(); for g in A3 A4 A5 A6; do for s in 40 41 42 43 44; do all+=("$g $s"); done; done
gpu_jobs=(); cpu_jobs=(); i=0
for j in "${all[@]}"; do
  if [ $i -eq 0 ] || [ $i -eq 7 ] || [ $i -eq 14 ]; then gpu_jobs+=("$j"); else cpu_jobs+=("$j"); fi
  i=$((i+1))
done

run_pool() {   # $1=device  $2=parallel  其余=作业
  local device="$1" par="$2"; shift 2
  printf '%s\n' "$@" | xargs -P "$par" -n 2 bash -c '
       dev="$0"; g="$1"; s="$2"
       OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
         "$PY" cleanmarl/run_sweep.py --config "$CONFIG" --group "$g" --seeds "$s" --device "$dev" \
         > "logs/sweeplogs/${g}_${s}.log" 2>&1 \
         && echo "  [ok $dev] $g seed$s" \
         || echo "  [FAIL $dev] $g seed$s -> logs/sweeplogs/${g}_${s}.log"
     ' "$device"
}

echo "[hybrid] GPU(单进程, 3作业): ${gpu_jobs[*]}"
echo "[hybrid] CPU(7并行, 17作业): ${cpu_jobs[*]}"
START=$(date +%s)
run_pool cuda 1 "${gpu_jobs[@]}" & GPU_PID=$!
run_pool cpu  7 "${cpu_jobs[@]}" & CPU_PID=$!
wait "$GPU_PID" "$CPU_PID"
echo "[hybrid] 全部完成, 用时 $(( $(date +%s) - START )) 秒"
