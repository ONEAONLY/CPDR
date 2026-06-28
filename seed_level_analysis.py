"""种子级(聚类)分析: 每个种子取64条eval均值作为1个重复单位, 在种子层面做配对检验.
这才是诚实的统计 (episode 级配对把64条相关episode当独立样本, p值虚低).
修种子后同seed各机制共享网络初始化 -> per-seed 差是纯机制效应.
"""
import glob, os, numpy as np, pandas as pd
from scipy import stats
base = "/home/ylj/CPDRE/logs/cleanmarl"
METRICS = ["sum_system_profit", "mean_shortage_rate", "own1", "ordN"]

def per_seed(group):
    rows = {}
    for r in sorted(glob.glob(f"{base}/{group}_*")):
        seed = int(os.path.basename(r).split("seed")[-1])
        d = pd.read_csv(os.path.join(r, "episode_summary.csv")); d = d[d.phase == "eval"]
        rows[seed] = dict(
            profit=d["sum_system_profit"].mean(),
            SR=d["mean_shortage_rate"].mean(),
            SR1=(1 - d["mean_own_service_rate_u1"]).mean(),
            SRN=(1 - d[["mean_own_service_rate_u2", "mean_own_service_rate_u3"]].mean(axis=1)).mean(),
        )
    return rows

groups = ["B1", "B2", "B3", "B4", "B5"]
P = {g: per_seed(g) for g in groups}
seeds = sorted(set.intersection(*[set(P[g]) for g in groups]))
n = len(seeds)
print(f"=== 种子级 (n={n} 个种子: {seeds}) ===\n")
print(f"{'组':<4}{'系统利润':>16}{'缺货SR':>14}{'SR_1':>10}{'SR_N':>10}")
for g in groups:
    pr = [P[g][s]['profit'] for s in seeds]; sr = [P[g][s]['SR'] for s in seeds]
    s1 = [P[g][s]['SR1'] for s in seeds]; sn = [P[g][s]['SRN'] for s in seeds]
    print(f"{g:<4}{np.mean(pr):>9.1f}±{np.std(pr,ddof=1):>5.1f}{np.mean(sr):>9.3f}±{np.std(sr,ddof=1):>4.3f}{np.mean(s1):>10.3f}{np.mean(sn):>10.3f}")

def paired(a, b, key='profit'):
    va = np.array([P[a][s][key] for s in seeds]); vb = np.array([P[b][s][key] for s in seeds])
    diff = va - vb; md = diff.mean(); sd = diff.std(ddof=1)
    se = sd / np.sqrt(n); tcrit = stats.t.ppf(0.975, n - 1)
    ci = (md - tcrit * se, md + tcrit * se)
    t, p = stats.ttest_rel(va, vb)
    allpos = (diff > 0).all() or (diff < 0).all()
    return md, ci, p, allpos, diff

print(f"\n=== 种子级配对 t 检验 (系统利润) ===")
print(f"{'对比':<10}{'均值差':>9}{'95% CI':>20}{'p':>9}{'全同号':>8}")
for a, b in [("B4", "B2"), ("B4", "B3"), ("B4", "B5"), ("B4", "B1")]:
    md, ci, p, ap, diff = paired(a, b)
    flag = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"{a} vs {b:<5}{md:>+9.2f}  [{ci[0]:>+6.2f},{ci[1]:>+6.2f}]{p:>9.4f}  {('是' if ap else '否'):>6} {flag}")
print("\n逐种子 B4-B2 差:", np.round(paired('B4','B2')[4], 2))
