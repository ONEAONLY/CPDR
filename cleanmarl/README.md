# CleanMARL — CPDRE 煤电直接互惠实验框架

CleanMARL 是本仓库自研的轻量级多智能体强化学习实现:绕开 RLlib,用 PyTorch 原生管理采样、GAE、
PPO/HAPPO 更新、GRU actor、集中 critic、**回报归一化(ValueNorm)**、日志与 checkpoint。服务于煤电
供应链 **直接互惠(CPDRE)** 研究。

> **权威规格 = `model/0626.md`**(尤其 §4.9 实施修订记录,记录本框架相对正文的全部落地改动)。

---

## 1. 运行前提

- **从仓库根目录运行**(`run_sweep.py` 自动把仓库根加入 `sys.path`;`--config` 路径相对 `cleanmarl/` 解析,写 `configs/xxx.yaml` 即可)。
- 依赖见仓库根 `requirements.txt`:`torch / numpy==1.24.3 / scipy / pandas / pyyaml`(cleanmarl 不走 RLlib,无需 ray/gym)。`pip install -r requirements.txt`(torch 按服务器 CUDA 选)。
- 本机(RTX4060)解释器 `/home/ylj/miniconda3/envs/marllib_4060/bin/python`;换机器见根 `CLAUDE.md`。下文 `python` 代指它。有 GPU 用 `--device cuda`,否则 `--device cpu`。

---

## 2. 实验总览

| 实验 | 目的 | 组别 | 入口 |
| :-- | :-- | :-- | :-- |
| **实验一** | 选主算法(无互惠) | A1–A6 | `--experiment exp1` |
| **实验二** | 直接互惠主实验(头条 B4 vs B2 / B4 vs B5) | B1–**B5** | `--experiment exp2` |
| **实验三** | 机制来源消融 | C1–C7 | `--experiment exp3` |
| 稀缺压力 | K 调小看互惠条件价值 | B2/B3/B4 | `configs/happo_cpdre_K30.yaml` |
| **非稳态** | 旺季时间/时长/幅度逐年随机(动态 vs 静态合同) | B1–B5 / C1–C7 | `configs/happo_cpdre_nonstat.yaml` |

**顺序:实验一 → 实验二 → 实验三**;非稳态在稳态主线确立后做(揭示动态互惠的条件价值)。
零参数启动器在 `cleanmarl/_run_*.sh`(已设单线程确定性 + 多核并行),例:
```bash
bash cleanmarl/_run_exp2_seeded.sh   # 实验二 B1-B5 x 12种子, 确定性
bash cleanmarl/_run_nonstat.sh       # 非稳态 B1-B5 x 12种子
bash cleanmarl/_run_exp3.sh          # 实验三 C1-C7
```

---

## 3. 实验一:算法选择

```bash
python cleanmarl/run_sweep.py --config configs/happo_cpdre.yaml --experiment exp1 --seeds 40,41,42,43,44
python cleanmarl/analyze_experiments.py --experiment exp1
```
- **A1/A2 = 派生式 base-stock 基线**(produce-to-forecast 煤企 θ + order-up-to 电企 ω;A1 无安全库存 / A2 含报童式安全库存,`z=Φ⁻¹(c_rep/(c_rep+h))`)。**不在评估集调参,无泄漏**。规则基线现以 **b2 执行**(煤企用动作里的派生 θ),与学习组共用同一需求与管线。
- A3=IPPO、A4=MAPPO、**A5=HAPPO**、A6=HAPPO+低扰动。
- **选法**:HAPPO 按**跨种子 reward_std 最小(最稳)**入选;它与 MAPPO 的利润差很小、在噪声内——选的是稳定性。
- 退出标准:学习型在**利润**上稳超规则基线(强规则 A2 的安全库存使其缺货更低,诚实权衡)。

---

## 4. 实验二:直接互惠机制主实验(核心)

```bash
bash cleanmarl/_run_exp2_seeded.sh    # B1-B5 x 12种子(推荐: 确定性)
python cleanmarl/analyze_experiments.py --experiment exp2   # 注: 用 episode 级检验, 见 §7 警示
python seed_level_analysis.py          # 诚实的种子级配对检验(正式结论用这个)
```

| 组 | 名称 | θ | λ₁(对 U1)| 记忆 μ | ημg |
| :-- | :-- | :-- | :-- | :-- | :-- |
| B1 | 派生公平规则基准 | 派生 base-stock | ≡1 | 无 | 无 |
| B2 | **非关系动态协调(头条对照)** | 学 | ≡1 | 无 | 无 |
| B3 | 规则互惠 | 学 | 依压力 χ̂ 触发 w_high | 无 | 无 |
| B4 | **关系专属动态直接互惠** | 学 | 学∈[1,λmax] | 用 | **有** |
| **B5** | **双边静态长协**(对照真实电煤长协) | 学 | 静态:淡季 U1 囤煤 ω_high + 旺季 λ=w_high | 无 | 无 |

- **两个头条对照**:**B4 vs B2**(关系互惠相对一般动态协调的增量)、**B4 vs B5**(动态相对静态合同)。
- **判读(spec §4.6.3 + 实测)**:稳态下系统利润 **B4≈B2≈B3≈B5(null)**——供给固定、再分配零和,互惠不增系统利润;但 **B3/B4 显著降低 U1 缺货 SR_1**(伙伴再分配,代价 SR_N 略升)。**头条指标应看伙伴供应可靠性 / 抗冲击鲁棒性,而非系统利润。**
- 若 B4 只压 SR_1 却抬 SR_N → 是对 U1 的优先权而非系统协调(表已并列 SR_1/SR_N)。

---

## 5. 实验三:机制来源消融

```bash
bash cleanmarl/_run_exp3.sh
python cleanmarl/analyze_experiments.py --experiment exp3   # + seed_level 检验
```

| 组 | 消融 | 识别目的 |
| :-- | :-- | :-- |
| C1 | 完整 B4 | 基准 |
| C2 | `g^U≡0`(disable_g_u) | 切断承购建信任 |
| C3 | `λ₁≡1`(fixed_lambda) | 动态保供权重是否必要 |
| C4 | 冻结 μ(freeze_memory) | 信任是否参与策略 |
| C5 | 冻结 θ(freeze_theta) | 仅配给能否解释 B4 |
| C6 | 信任不进观测(no_recip_in_obs) | 是否真用关系信息 |
| **C7** | **η=0**(no_recip_reward) | 改善是否非奖励塑形 |

- 稳态消融:**C5 冻结 θ 利润崩** → 系统价值压倒性来自**可学习产能路径 θ**;C7≈C1 且去 ememory 不损 → 价值非奖励塑形。
- **判读指标在非稳态下换成 SR_1 / 稳健性**:看去掉哪个部件会让 B4 的"抗非稳态伙伴保护"像 B5 一样崩。
- ⚠️ **C5 注意**:`freeze_theta` 用固定季节排程 θ_offpeak/θ_peak。该排程已由反向坏值 1.15/0.80 修正为 **0.95/1.20(旺季多产)**;旧 C5 结果作废,须用新排程重跑。

---

## 6. 非稳态季节(动态 vs 静态合同 的关键检验)

```bash
bash cleanmarl/_run_nonstat.sh                 # B1-B5 x 12种子, nonstationary_season=true
python nonstat_cmp.py                           # 稳态 vs 非稳态 种子级对比
```
- 每年随机抽 `φ_D~U[6,22]`、`W_on~U[16,34]`、`a_D~U[0.30,0.55]`,仍 mean-preserving(年均=1)。
- **关键不对称**:需求按真实随机季节走;**长协 B5 / 规则按名义日历(φ=13,W_on=26)提前签死**(对应真实长协月度量分解),真实季节偏离时静态合同错配;**B2/B4 凭实现需求 + 压力 χ̂ 适应**。双方均观测当下真实季节(现实合理),差距来源是"日历承诺合同 vs 逐周再优化",非信息不对称(spec §4.9.1)。
- **实测核心发现**:稳态下 B5 与 B4 都能保住 U1;**非稳态下 B5 的 U1 保护显著崩溃(SR_1 0.010→0.036),而 B4 稳健不变(0.009)**,B4 vs B5 SR_1 差 p<0.001。→ **直接互惠的独特价值 = 不确定环境下对伙伴提供静态合同给不了的稳健供应保障。**

---

## 7. 统计口径(★重要,易踩的坑)

- **固定评估轨迹**:`env.eval_reset(idx)` 内部 `rng=default_rng(9000+idx)`,与训练种子无关 → 任何组别/种子都在同一批 64 条固定轨迹上评估,可配对。非稳态下该 idx 的随机季节也被钉死、各组一致。
- **训练确定性**:`run_one` 调 `set_seed(seed)` 固定 torch+numpy → 同种子下各机制组**共享网络初始化**,机制差异被干净隔离、可复现(`OMP_NUM_THREADS=1` 时完全确定)。**未固定种子时 run 间噪声 ±13/种子,会淹没 ~6 的机制效应。**
- **种子级 vs episode 级(关键)**:`analyze_experiments.py` 内置的是 **(seed×episode) 配对** Wilcoxon——把同一模型下相关的 64 条 episode 当独立样本,**伪重复、p 值虚低若干数量级**。**正式结论必须用种子级检验**(每种子取 64 条 eval 均值作 1 个重复单位,`seed_level_analysis.py` / `stress_cmp.py` / `nonstat_cmp.py`)。主实验种子数用 **12**(40–51),不是 spec 早期写的 5。

---

## 8. 回报归一化(ValueNorm)

奖励被近乎恒定的利润地板(~328±8)主导,critic 直接回归原始回报会被大常数淹没 → `vf_explained_var≈0`、
advantage 无信号。`cleanmarl/algos/value_norm.py`(PopArt 式)让 critic 在归一化空间回归、GAE/bootstrap
在原始空间,实测 `vf_explained_var` 由 ≈0 升至 ~0.8。开关 `use_value_norm`(yaml,默认开)。**这是 B4 在
修复统计口径后能与 B2 公平比较的前提。**

---

## 9. 组别 → 环境映射(`run_sweep.GROUP_ENV_MAP`)

| 组 | mechanism | allocation | 算法 | 备注 |
| :-- | :-- | :-- | :-- | :-- |
| A1/A2 | **b2** | fair | rule_a1/a2 | 派生 base-stock(A1 无安全 / A2 含安全),煤企用动作派生 θ |
| A3–A6 | b2 | fair | ippo/mappo/happo(A6 低扰动) | 学习无互惠,学 θ |
| B1 | **b2** | fair | rule_b1 | = A2 派生 base-stock(含安全库存) |
| B2 | b2 | weighted | (YAML) | 非关系动态协调,λ=1,无记忆,无 ememory |
| B3 | b3 | weighted | (YAML) | 规则互惠:学 θ + λ 依 χ̂ 触发 |
| B4 | b4 | weighted | (YAML) | 完整关系互惠,学 θ 和 λ,含 ememory |
| **B5** | **long_contract** | weighted | (YAML) | 双边静态长协:学 θ + 静态 ω/λ(名义日历),无互惠 |
| C1–C6 | b4 变体 | weighted | (YAML) | C2 disable_g_u / C3 fixed_lambda / C4 freeze_memory / C5 freeze_theta / C6 no_recip_in_obs |
| C7 | b4 | weighted | (YAML) | η=0(no_recip_reward)|

`--group` 自动绑定机制;规则组用 `force_algo` 强制规则策略,`--config` 取哪个 yaml 都行(只用它的环境+超参)。
单组 `--group B4`;单种子 `--seeds 42`;冒烟 `--timesteps 1560`(只验管线,数值无意义)。

---

## 10. 代码结构

```text
cleanmarl/
├── algos/{happo,mappo,ippo,rule_policy}.py   # 算法 + 规则基线
├── algos/value_norm.py                       # ValueNorm 回报归一化(PopArt 式)
├── configs/{happo,mappo,ippo,default}_cpdre.yaml + happo_cpdre_{K30,nonstat}.yaml
├── core/{trainer,logger,episode_logger}.py
├── models/models.py                          # GRUActor / CentralizedCritic
├── run_sweep.py        # 多种子/组别扫描 + --experiment 预设 + GROUP_ENV_MAP + set_seed
├── analyze_experiments.py  # 组间表 + 配对检验(episode 级, 见 §7 警示)
└── _run_*.sh           # 零参数确定性并行启动器
```
环境本体:`custom_envs/coal_power_direct_reciprocity_env.py`。
根目录 `seed_level_analysis.py / stress_cmp.py / nonstat_cmp.py` 为**种子级诚实分析器**。

---

## 11. 已知缺口

- `analyze_experiments.py` 仍是 episode 级检验(p 值虚低),正式结论用种子级分析器(§7)。建议后续把种子级并入。
- C5 旧结果用反向坏 θ 排程跑的,须用修正排程(0.95/1.20)重跑。
- 实验四/五/六(参数敏感性 / 规模范围 / 真实数据)未在 `run_sweep` 做一键预设。
- 中途评估不触发(`eval_freq` 与 rollout 步数对不上),只有 final eval 写 `phase=eval`;中途中断的 run 无 eval 行(白跑)。
- 多互惠 `h>1`、多电企 `m>3` 的观测/日志仍有 u1/u2/u3、pad3 等固定写法,扩展前需改 schema。
- B5 的 long_contract 已移出互惠模式集合,g/μ 恒 0(行为零影响,已验证)。

---

## 12. 阶段性结论(供论文)

- 学习型稳超规则基线(利润);系统稳定性的决定因素是**可学习的产能路径 θ**(C5 冻结即崩)。
- **稳态、可签约环境**:动态直接互惠在系统利润上 ≈ 非关系协调 ≈ 静态长协(null);互惠是**伙伴专属的供应安全再分配**,非系统效率改进——学出的动态互惠收敛到最优静态合同。
- **非稳态环境**:动态互惠**稳健保护伙伴**,静态长协因日历错配而保护崩溃 → 直接互惠的独特价值在**合同失效地带**(不确定/冲击/背叛/多伙伴)。头条指标:伙伴供应可靠性 / 抗冲击鲁棒性。
