# CLAUDE.md

本仓库是煤电供应链**直接互惠**(CPDRE)的多智能体强化学习(MARL)研究代码。
**权威规格现为 `model/0626.md`**(取代旧 0611);一切代码以它为准对齐。代码已按 0626 落地:
价格固定为 1、平台型季节(`_seasonal_factor`)、互惠效用 ημg 进主模型(`use_reciprocity_reward=True`)、
g^C 用基准产能公平基准、g^U 用剩余供给减少、K=1.2·m·D̄、γ=0.99、c_rep=1.70。0608/0611/0623/0624/0625
为历史版本;`model/0625_issues.md` 记录了 0625→0626 的修订清单。

**核心研究问题**:关系专属动态直接互惠(B4)能否优于非关系型动态协调(B2)?
即区分"一般状态感知协调 / 关系专属互惠 / 单纯配给偏置"三类机制来源。头条对照 **B4 vs B2**
(B2 已是会动态学 θ 的非关系协调者);C7(η=0)vs B2 排除奖励塑形,C5(仅配给)查配给偏置。

只用项目根下的自研框架 **`cleanmarl/`**(PyTorch 原生:采样 + GAE + PPO/HAPPO 更新 +
GRU + 集中 critic)。当前主线:1 个煤企 + 3 个电企 + 1 个互惠电企(U1)的 B4 实验。

> **本文件是跨机器/跨会话的唯一可靠交接渠道。** Claude Code 的"本机记忆"
> (`~/.claude/...`)和对话历史**不随项目走**;只有提交进 git 的东西(本文件、`model/`
> 规格、代码、注释)才会跟到新机器。重要信息一律写进这里并提交。

---

## ⚙️ 运行环境(★换机器必须按本机改★)

代码在 **WSL Ubuntu** 文件系统里,对应 Windows 侧 `\\wsl.localhost\Ubuntu\...`,同一份文件。

**训练/分析必须用装了 torch 的 conda 环境**:
- **换电脑后**:conda 环境名 / Python 绝对路径 / 项目路径 / 是否有 WSL / 是否有 GPU 都可能不同。
  先确认 torch 装在哪个环境(`conda env list` + `python -c "import torch"`),把下面命令里的
  路径替换成你本机的。有 GPU 用 `device: cuda`,没有就 `--device cpu`。

### 本机配置(RTX 4060 机器,2026-06 实测跑通)
- 项目路径:**`/home/ylj/CPDRE`**(Windows 侧 `\\wsl.localhost\Ubuntu\home\ylj\CPDRE`)。
- conda 环境:**`marllib_4060`**,解释器
  `/home/ylj/miniconda3/envs/marllib_4060/bin/python`。
- 栈:Python 3.8.20 + torch 2.3.1(**CUDA 可用**)+ numpy 1.24.3 + scipy 1.10.1 + pandas 2.0.3。
  marllib 1.0.3 从 `/home/ylj/MARLlib` 可编辑安装(本框架不依赖它,训练不走 RLlib)。
- 已实测:`cleanmarl/smoke_test.py` 端到端通过,默认 `device=cuda`。
- 注:`requirements.txt` 写的 `gym>=0.21`/`ray[rllib]>=1.9` 高于本机实装(gym 0.20、ray 1.8),
  但 cleanmarl 是 PyTorch 原生、训练/分析不走 RLlib,所以不影响。

> 上一台机器(历史参考)是 `/home/asus/code/New_Marllib/MARLlib` + 环境 `marllib_torchtest`
> (torch 1.9.0+cu111)。换机器只需照上面"先确认 torch 在哪个环境"重做一遍。

本机命令模板(从 Windows 调 WSL;若已在 WSL 终端内则去掉 `wsl.exe -e bash -lc` 外壳):
```bash
wsl.exe -e bash -lc 'cd /home/ylj/CPDRE && \
  /home/ylj/miniconda3/envs/marllib_4060/bin/python cleanmarl/run_sweep.py ...'
```
不想写长路径就先 `conda activate marllib_4060` 再用 `python`。

---

## 实验流程(实验一 + 实验二)

目前只完成了实验一与实验二的代码，也可以自行检查代码的完善程度

入口:`cleanmarl/run_sweep.py`(跑实验)+ `cleanmarl/analyze_experiments.py`(出表)。
一条命令跑完一个实验的所有组别 × 所有种子,结果落在 `logs/cleanmarl/{组}_{算法}_seed{种子}/`
(每个 run 一个文件夹,含 `config.json / progress.csv / episode_summary.csv / run_meta.json`)。

### 实验一:基准策略与算法选择(选主算法)
```bash
python cleanmarl/run_sweep.py --config configs/happo_cpdre.yaml --experiment exp1 --seeds 40,41,42,43,44
python cleanmarl/analyze_experiments.py --experiment exp1
```
- A1/A2 **派生式 base-stock 基线**(A1 无安全库存 / A2 含 newsvendor 安全库存),A3=IPPO、
  A4=MAPPO、A5=HAPPO、A6=HAPPO+低需求扰动,均**无互惠**。规则基线不在评估集上调参(无泄漏)。
- 看 `system_profit` 高且跨种子 `reward_std` 小者 → 定主算法(通常 HAPPO)。
- 退出标准:学习型要稳定优于规则基线;否则先调环境/算法,别进实验二。
- **修正版结果(2026-06-27,统一需求口径后)**:A1=439.8/缺货0.074,A2=419.4/**0.008**,
  A3=437/0.059,A4=451/0.056,**A5 HAPPO=455.6/0.029(reward_std 7.8 最小)**,A6=451/0.054。
  判读:**学习赢利润(HAPPO 456>强规则 A2 419)、强规则赢缺货(A2 安全库存 0.008)**,诚实权衡;
  主算法仍选 **HAPPO**。之前"调优规则碾压 HAPPO"是需求口径 bug 假象,已消除。

### 实验二:直接互惠机制主实验(核心结论)
```bash
python cleanmarl/run_sweep.py --config configs/happo_cpdre.yaml --experiment exp2 --seeds 40,41,42,43,44
python cleanmarl/analyze_experiments.py --experiment exp2
```
- B1–B4(已删 B5)。**B4 vs B2 是头条对比**;关系专属性由实验三的 C7(η=0)vs B2 + C3/C6 联合识别。
- 分析脚本自动出组间表 + 按 (seed, episode) 配对的 **Wilcoxon + bootstrap CI**(B4 vs B2/B3/B1)。
- 判读(spec 0626 4.6.3):
  - B4≈B2 → 关系互惠不优于非关系动态协调(B2 本身就是动态协调者)。
  - B4>B2 **且 C7>B2**(去 ημg 仍优)、SR_N/J 未恶化 → 关系专属互惠有结构性价值(强结论)。
  - B4>B2 但 C7≈B2 → 价值由关系动机(ημg)使能,作诚实弱表述。
  - 若 B4 只改善 SR_1 却显著抬高 SR_N → **不能**算系统协调改善(表已并列 SR_1/SR_N 供审计)。

### 实验三:机制来源消融(可选)
```bash
python cleanmarl/run_sweep.py --experiment exp3 --seeds 40,41,42,43,44   # C1-C7
python cleanmarl/analyze_experiments.py --experiment exp3
```

### 常用参数
- `--group B4` 只跑单组;`--seeds 42` 单种子;`--timesteps 1560` 覆盖步数(冒烟,1 rollout)。
- analyze:`--groups B2,B3,B4` 覆盖默认组别;`--pairs B4:B2` 自定义配对;
  `--metric system_profit,SR` 选检验指标;`--csv out.csv` 导出汇总表。
- **冒烟测试(1 rollout)只验管线,数值无意义**;正式结论必须跑满 config 的
  `total_timesteps`(1e5)× 5 种子。

---

## 固定评估轨迹(为什么能做配对检验)

**训练随机性与评估随机性故意分开**:
- 训练种子(`--seeds 40..44`)→ `cfg['env']['seed']`,决定训练时的随机流。5 个种子 = 方法重复 5 次。
- 评估走 `env.eval_reset(episode_idx)`,内部固定 `rng = default_rng(9000 + episode_idx)`,
  **与训练种子无关**。所以不论哪个训练种子、哪个组别,评估都在**同一批 64 条固定轨迹**
  (episode 0..63 ↔ 种子 9000..9063)上跑,整条 156 周需求/价格/供给冲击被钉死。

→ `B4-seed40-ep5` 和 `B5-seed40-ep5` 面对**完全相同的需求实现**,差异只来自机制本身。
这正是 `analyze_experiments.py` 按 (seed, episode) 配对做 Wilcoxon 的前提,能消掉"剧本难易"
这一最大噪声源,大幅提升统计功效。

---

## 组别 → 环境机制映射(`run_sweep.GROUP_ENV_MAP`)

| 组 | mechanism | allocation | 算法/策略 | 备注 |
|---|---|---|---|---|
| A1/A2 | **b2** | fair | rule_a1/rule_a2 | **派生式 base-stock 基线**(A1 无安全库存 / A2 含 newsvendor 安全库存),θ/ω 从需求+成本推出,无魔法数。用 b2 让基线与学习组面对**同一需求**(★见下) |
| A3/A4/A5 | **b2** | fair | ippo/mappo/happo | 学习无互惠,煤企**学 θ**(产能路径可学) |
| A6 | **b2** | fair | happo + low_noise | 低扰动稳定性 |
| B1 | **b2** | fair | rule_b1 | 公平规则基准 = A2 派生 base-stock(含安全库存) |
| B2 | b2 | weighted | (YAML 算法) | **非关系动态协调基准(头条对照)**,λ=1,无记忆,无 ημg |
| B3 | **b3** | weighted | (YAML 算法) | 规则互惠:学 θ + λ 依 χ̂ 触发,无记忆 |
| B4 | b4 | **weighted** | (YAML 算法) | 完整关系专属互惠,学 θ 和 λ,**含 ημg 奖励** |
| C1–C6 | b4 变体 | weighted | (YAML 算法) | C2=disable_g_u, C3=fixed_lambda, C4=freeze_memory, C5=freeze_theta, C6=去关系观测 |
| C7 | b4 | weighted | (YAML 算法) | **η=0**(use_reciprocity_reward=False),去互惠效用反证 |

`--group A4` 等会**强制覆盖算法**(force_algo),所以 `--config` 用哪个 yaml 都行(只取它的环境+超参)。

---

## 关键环境语义(容易踩的坑)

环境:`custom_envs/coal_power_direct_reciprocity_env.py`。煤企动作 2D `(θ,λ)`,电企 1D `(ω)`。

1. **`none` 模式煤企不学 θ** —— θ 钉死成手设季节排程。只有 `b2/b3/b4/dynamic` 让煤企学 θ。
   所以"无互惠学习"组要用 **b2 不是 none**。**规则基线(A1/A2/B1)现也用 b2**(煤企用
   `RulePolicy` 算出的派生 θ 作动作,env 执行),不再用 none。
   ★**(2026-06-27 修复的大 bug)**:`_base_demand_vector` 历史上给 `none/long_contract/trigger`
   用遗留两档需求 `demand_offpeak/demand_peak`=1.0/1.4,而 b 模式用平台季节 0.8/1.2,相差 **1.25×**。
   这让规则基线(none)与学习组(b2)**面对不同需求剧本**,所有"基线 vs 学习"对比(实验一/二)
   全被污染,且破坏固定评估轨迹的配对前提。已统一:**所有模式都用平台 `D̄·s^D_t`**。修复后
   同 θ/ω 下 none 与 b2 逐周需求与利润完全一致。**A1/A2 旧结果作废已重跑;A3–A6(b2)需求口径
   本就对、无需重跑。**
1b. **派生式原理基线(替代旧手设/网格)** —— A1/A2/B1 不再用手拍的 θ 排程+魔法数 ω=2.5。
   煤企 `env.baseline_coal_theta`:θ·K 覆盖当期预期需求 `m·D̄·s^D_t`(+A2 安全缓冲 z·CV);
   电企 `env.baseline_power_omega`:order-up-to 覆盖 = lead+1(+A2 安全 z·CV·√(lead+1))。
   `z=Φ⁻¹(c_rep/(c_rep+h))≈2.11`(临界分位),CV≈demand_sigma。**全从需求/成本推出,不在评估集上
   拟合(无测试集泄漏)**。旧基线 θ_peak=0.80(旺季减产)是反的,缺货 0.187 → 派生 A2 缺货 0.008。
2. **b4/b3 的 λ 必须配 `allocation_mode: weighted`** —— 用 fair 会被 `_force_fair_allocation()`
   把权重重置为 1,保供闭环断掉。none/b2 强制 fair(λ=1)。
3. **电企利润用真实需求**:`r·D − p·Y − c_rep·S − h·I`(外部应急补货满足当期发电),不是 `r·served`。
4. **互惠效用奖励 ημg 已在主模型(0626)**(`use_reciprocity_reward=True`,仅对 b4 生效):
   煤企奖励 = 归一化利润 **+ η_C·μ^C·g^C** − λ_S·缺煤 − λ_J·(1−Jain),U1 另加 η_U·μ^U·g^U。
   C7 把开关置 False 做"非奖励塑形"反证。判机制差异仍看奖励外的独立指标(SR/SR_N/J/利润)。
5. **SR 派生**:`service_rate=(D−S)/D` 是满足率(越大越好);分析里 `SR_i = 1 − own_service_rate_ui`,
   系统 `SR = mean_shortage_rate`。
6. **异质动作**:`act_dims=[2,1,1,1]`,buffer pad 到 `max_act_dim=2`;log_prob 按动作维求和成标量,
   **不破坏 HAPPO** 顺序更新。

---

## 决策背景(为什么是 cleanmarl;一个要注意的现象)

- **为什么自研 cleanmarl**:旧的 MARLlib/RLlib 里 **HAPPO 价值网络学不动**
  (`vf_explained_var≈0`,各 agent 价值函数完全不学)。试过改 optimizer 绑定/手动更新 critic 等
  多种修法都失败(崩溃或报 grad 错)。根因是 HAPPO 的逐 agent 异构更新和 RLlib 优化流程冲突。
  于是绕开 RLlib,用 PyTorch 原生重写 `cleanmarl/`,HAPPO 价值学习恢复正常。关键设计:
  critic 独立优化器 + 每 agent 独立价值输出、按 episode 重组数据(RNN 隐藏态 episode 内连续/
  间重置)、loss 直接 backward、HAPPO 顺序更新(更新后重评 ratio 迭代后续 advantage)。
  **结论:用 CPDRE 跑 HAPPO 一律用 cleanmarl,别用 MARLlib 的 HAPPO。**
- **要注意的现象:`vf_explained_var ≈ 0` 不一定是 bug。** 对齐 spec 后奖励是"绝对利润量",
  被一大坨**几乎恒定的基础利润**主导(reward ~328±8,相对波动才 ~2.5%)。于是 `value_loss`
  能降(学会了那个大常数量级),但 `vf_explained_var` 上不去(可被状态解释的方差极小,剩下大多是
  不可预测的需求噪声)。后果:advantage 信号弱、训练曲线偏平。**判断机制差异要看组间
  `shortage_rate / system_profit / SR_1-SR_N` 的差(跑满 5 种子后),不是单组训练曲线。**
  若以后想让 RL 学得更动,杠杆是奖励/价值归一化(running mean-std 标准化回报,或 PopArt,
  或把利润项中心化),把"大常数地板"减掉。

---

## 文件地图

- `model/0611(1).md` —— 权威规格(**勿改**)。`model/cleanmarl_0611_issues.md` 代码差距清单;
  `model/0611_remaining_issues.md` 文档自身待完善点。
- `cleanmarl/run_sweep.py` —— 多种子/多组别扫描 + `--experiment` 预设 + `GROUP_ENV_MAP` + `run_meta.json`。
- `cleanmarl/analyze_experiments.py` —— 出实验一/二/三对比表 + 配对 Wilcoxon/bootstrap。
- `cleanmarl/train.py` —— 单次训练入口(`build_policy` 路由 happo/mappo/ippo + rule_*)。
- `cleanmarl/algos/{happo,mappo,ippo}.py` —— 算法;`rule_policy.py` 规则/固定基线(无梯度)。
- `cleanmarl/models/models.py` —— GRUActor + CentralizedCritic(支持 `rnn_layers`)。
- `cleanmarl/core/{trainer,episode_logger,logger}.py` —— 训练基类 + 每 episode 指标(`episode_summary.csv`,
  MEAN/SUM/STD_FIELDS)+ 训练进度(`progress.csv`)。
- `cleanmarl/configs/happo_cpdre.yaml` —— 主配置(对齐 spec 表4-6:γ0.99, lr1e-4, rnn_layers2,
  chunk_len10, eval_episodes64)。mappo/ippo 同构。`default.yaml` 是 train.py 的默认。
- `custom_envs/coal_power_direct_reciprocity_env.py` —— CPDRE 环境本体。
- 产物:`logs/cleanmarl/<run>/`、`checkpoints/`。

---

## 已知尚未完成(按需推进)

- **#4** 连续动作未 squash(采样可能越界,clip 后执行但用 clip 前动作算 log_prob)。
- **#6** HAPPO 顺序更新只修相邻 agent,非标准 Kuba 2022(已声明不依赖单调性)。
- **#9/#10** h>1 完全向量化、m>3 obs 不截断 —— 扩展实验(E 系列)才需要。
- **中途评估从不触发**:trainer.py 用 `current_step % eval_freq == 0`,而 step 永远是
  rollout_length 的整数倍,跟 eval_freq 对不上 → 只有训练结束时的 final eval 写 `phase=eval`。
  后果:中途 Ctrl+C 的 run 一行 eval 都没有(白跑)。要改成"距上次 eval ≥ eval_freq 就评估"
  + 中断时也补跑 final eval。
- 文档侧:训练并行环境数/rollout 步数未交代、煤价固定、λ_S/λ_J 取值来源等
  (见 `model/0611_remaining_issues.md`)。

---

## 验证现状

**0626 落地后(2026-06)** B4/B3/C7 训练管线 1-rollout 冒烟全部 exit=0、checkpoint 正常。
环境机制冒烟(随机动作整条 episode)确认:平台季节(旺季平台 1.2/淡季 0.8、年均 1.0)、价格恒 1、
χ 两边穿越(宽松≈48%/紧张≈52%)、g^U/g^C 真实点火(B4 训练日志 mean_g_u≈0.42、mean_g_c≈0.10、
μ_c→0.74、λ1≈1.48)、b2 干净(Σ|g|=0、λ≡1)、K=3.6 旺季 θ^base=1.0<1.2 有余量。
**正式结论仍需跑满 1e5 步 × 5 种子**(冒烟为随机动作,只验机制可发生,非互惠均衡)。
