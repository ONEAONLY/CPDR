import glob, os, numpy as np, pandas as pd
from scipy import stats

def per_seed(base, g):
    out = {}
    for r in sorted(glob.glob(f"{base}/{g}_*")):
        s = int(os.path.basename(r).split("seed")[-1])
        d = pd.read_csv(os.path.join(r, "episode_summary.csv")); d = d[d.phase == "eval"]
        out[s] = dict(
            profit=d["sum_system_profit"].mean(), SR=d["mean_shortage_rate"].mean(),
            SR1=(1 - d["mean_own_service_rate_u1"]).mean(),
            SRN=(1 - d[["mean_own_service_rate_u2", "mean_own_service_rate_u3"]].mean(axis=1)).mean())
    return out

def ttest(P, seeds, a, b, k):
    va = np.array([P[a][s][k] for s in seeds]); vb = np.array([P[b][s][k] for s in seeds]); d = va - vb
    t, p = stats.ttest_rel(va, vb); se = d.std(ddof=1) / np.sqrt(len(d)); tc = stats.t.ppf(.975, len(d) - 1)
    f = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
    return f"差={d.mean():+.2f} CI[{d.mean()-tc*se:+.2f},{d.mean()+tc*se:+.2f}] p={p:.3f} {f}"

def block(base, label):
    P = {g: per_seed(base, g) for g in ["B1", "B2", "B3", "B4", "B5"]}
    seeds = sorted(set.intersection(*[set(P[g]) for g in ["B2", "B3", "B4", "B5"]]))
    print(f"=== {label} (n={len(seeds)}) ===")
    for g in ["B1", "B2", "B3", "B4", "B5"]:
        if g not in P or not P[g]: continue
        pr = [P[g][s]["profit"] for s in seeds]; sr = [P[g][s]["SR"] for s in seeds]
        s1 = [P[g][s]["SR1"] for s in seeds]; sn = [P[g][s]["SRN"] for s in seeds]
        print(f"  {g}: 利润 {np.mean(pr):6.1f}±{np.std(pr,ddof=1):4.1f}  总缺货 {np.mean(sr):.3f}  SR_1 {np.mean(s1):.3f}  SR_N {np.mean(sn):.3f}")
    print(f"  B4 vs B2 利润: {ttest(P,seeds,'B4','B2','profit')}")
    print(f"  B4 vs B5 利润: {ttest(P,seeds,'B4','B5','profit')}")
    print(f"  B2 vs B5 利润: {ttest(P,seeds,'B2','B5','profit')}  <- 动态(B2) vs 错配静态长协(B5)")
    print(f"  B4 vs B5 总缺货: {ttest(P,seeds,'B4','B5','SR')}")
    print()

block("/home/ylj/CPDRE/logs/cleanmarl", "稳态 (基线)")
block("/home/ylj/CPDRE/logs/cleanmarl_nonstat", "非稳态 (旺季时间/时长/幅度逐年随机)")
