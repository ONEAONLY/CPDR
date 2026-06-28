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
    tt, p = stats.ttest_rel(va, vb); se = d.std(ddof=1) / np.sqrt(len(d)); tc = stats.t.ppf(.975, len(d) - 1)
    f = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
    return f"差={d.mean():+.2f} CI[{d.mean()-tc*se:+.2f},{d.mean()+tc*se:+.2f}] p={p:.3f} {f}"

def block(base, label):
    P = {g: per_seed(base, g) for g in ["B2", "B3", "B4"]}
    seeds = sorted(set(P["B2"]) & set(P["B4"]) & set(P["B3"]))
    print(f"=== {label} (n={len(seeds)}) ===")
    for g in ["B2", "B3", "B4"]:
        pr = [P[g][s]["profit"] for s in seeds]; sr = [P[g][s]["SR"] for s in seeds]
        s1 = [P[g][s]["SR1"] for s in seeds]; sn = [P[g][s]["SRN"] for s in seeds]
        print(f"  {g}: 利润 {np.mean(pr):6.1f}±{np.std(pr,ddof=1):4.1f}  总缺货 {np.mean(sr):.3f}  SR_1 {np.mean(s1):.3f}  SR_N {np.mean(sn):.3f}")
    print(f"  B4 vs B2 利润  : {ttest(P,seeds,'B4','B2','profit')}")
    print(f"  B4 vs B2 总缺货: {ttest(P,seeds,'B4','B2','SR')}")
    print(f"  B4 vs B2 SR_1  : {ttest(P,seeds,'B4','B2','SR1')}")
    print()

block("/home/ylj/CPDRE/logs/cleanmarl", "K=3.6 宽松 (基线)")
block("/home/ylj/CPDRE/logs/cleanmarl_K30", "K=3.0 稀缺 (压力)")
