# CleanMARL — CPDRE 煤电直接互惠实验框架

CleanMARL 是本仓库自研的轻量级多智能体强化学习实现:绕开 RLlib 训练循环,用 PyTorch 原生管理
采样、GAE、PPO/HAPPO 更新、GRU actor/critic、集中 critic、日志与 checkpoint。它服务于煤电供应链
**直接互惠(CPDRE)** 研究。

> **权威规格 = `model/0626.md`**(取代旧 0611)。环境与配置已按 0626 落地:煤价固定为 1、平台型
> 季节、互惠效用 `η·μ·g` 进主模型、`g^C` 用基准产能公平基准、`g^U` 用剩余供给减少、`K=1.2·m·D̄`、
> `γ=0.99`、`c_rep=1.70`。主线:1 煤企 + 3 电企(其中 U1 为互惠电企)。

---

## 1. 运行前提

- **从仓库根目录运行**(`run_sweep.py` 会自动把仓库根加入 `sys.path`;`--config` 路径相对 `cleanmarl/` 解析,写 `configs/xxx.yaml` 即可)。
- 需要装了 torch 的 Python 环境。**本机(RTX 4060)**:conda 环境 `marllib_4060`,解释器
  `/home/ylj/miniconda3/envs/marllib_4060/bin/python`(换机器见仓库根 `CLAUDE.md`)。
- 依赖:`torch / numpy / gym / pyyaml`(`ray` 仅因环境基类 import,训练不走 RLlib)。

下文命令用 `python` 代指上面的解释器。有 GPU 用 `--device cuda`,没有用 `--device cpu`。

---

## 2. 实验总览:六个实验,逐级递进

| 实验 | 目的 | 组别 | run_sweep 预设 |
| :-- | :-- | :-- | :-- |
| **实验一** | 环境可训练性 + 选主算法(无互惠) | A1–A6 | `--experiment exp1` ✅ |
| **实验二** | 直接互惠主实验(**头条 B4 vs B2**) | B1–B4 | `--experiment exp2` ✅ |
| **实验三** | 机制来源消融(**C7 防自导自演**) | C1–C7 | `--experiment exp3` ✅ |
| 实验四 | 参数敏感性与稳健性 | 扫 K/ρ_max/σ_D/η/… | 手动覆盖参数(见 §6) |
| 实验五 | 规模 m / 互惠范围 h / 竞争 ρ | — | 手动覆盖参数(见 §6) |
| 实验六 | 真实数据外部回放验证 | 冻结策略回放 | 单独脚本(见 §6) |

**强制顺序:实验一 → 实验二 → 实验三。** 实验一不过(学习型不稳定优于规则基线)就**不要**进实验二——
否则会把"算法训练失败"误判成"互惠机制无效"。实验四/五/六是稳健性/边界/外部效度,主结论(exp1–3)定了再做。

每个实验 = 一条 `run_sweep` 命令(跑完该实验所有组别 × 所有种子)+ 一条 `analyze` 命令(出表 + 配对检验)。
结果落在 `logs/cleanmarl/{组}_{算法}_seed{种子}/`。

---

## 3. 实验一:环境可训练性与算法选择

```bash
python cleanmarl/run_sweep.py --config configs/happo_cpdre.yaml --experiment exp1 --seeds 40,41,42,43,44
python cleanmarl/analyze_experiments.py --experiment exp1
```

- **A1/A2** 规则基线(固定 / 状态感知 base-stock,煤企固定季节排程 θ);**A3=IPPO、A4=MAPPO、
  A5=HAPPO、A6=HAPPO+低扰动**,均无互惠(`mechanism_mode` 用 `b2`,煤企**学 θ**)。
- **看什么**:`system_profit` 高、跨种子 `reward_std` 小、`shortage_rate` 低、训练稳定 → 定主算法
  (通常 HAPPO)。
- **退出标准**:学习型(A3–A6)要稳定优于规则基线(A1/A2)。否则先调环境/算法,别进实验二。

---

## 4. 实验二:直接互惠机制主实验(核心)

```bash
python cleanmarl/run_sweep.py --config configs/happo_cpdre.yaml --experiment exp2 --seeds 40,41,42,43,44
python cleanmarl/analyze_experiments.py --experiment exp2
```

| 组 | 名称 | θ | λ₁ | 记忆 μ | 互惠奖励 ημg |
| :-- | :-- | :-- | :-- | :-- | :-- |
| B1 | 公平规则基准 | 季节排程 | ≡1 | 无 | 无 |
| B2 | **非关系动态协调基准(头条对照)** | 学 | ≡1 | 无 | 无 |
| B3 | 规则互惠 | 学 | 依 χ̂ 触发 | 无 | 无 |
| B4 | **关系专属动态直接互惠** | 学 | 学∈[1,λmax] | 用 | **有** |

- **头条对照 B4 vs B2**:B2 已是"会动态学 θ 的非关系协调者",所以 B4 相对它的改善**不能**归因为
  "单纯会动态协调"。analyze 自动做 (seed, episode) 配对 **Wilcoxon + bootstrap CI**(B4 vs B2/B3/B1)。
- **判读(spec 0626 §4.6.3)**:
  - `B4 ≈ B2` → 关系互惠不优于非关系动态协调。
  - `B4 > B2` **且** SR_N、公平 J 未恶化 → 关系专属互惠有价值(还需实验三 C7 坐实非奖励塑形)。
  - 若 `B4` 只压低 SR_1 却抬高 SR_N → **不算**系统协调,只是对 U1 的优先权(表已并列 SR_1/SR_N 供审计)。
- **算法稳健性补充**:可在 B4 机制下换 `--config configs/{ippo,mappo}_cpdre.yaml` 跑同组,检验结论不依赖单一算法。

---

## 5. 实验三:机制来源消融(防"自导自演")

```bash
python cleanmarl/run_sweep.py --config configs/happo_cpdre.yaml --experiment exp3 --seeds 40,41,42,43,44
python cleanmarl/analyze_experiments.py --experiment exp3
```

| 组 | 消融 | 识别目的 |
| :-- | :-- | :-- |
| C1 | 完整 B4 | 基准 |
| C2 | `g^U≡0` | 切断宽松期承购建信任 |
| C3 | `λ₁≡1` | 紧张期专属权重是否必要 |
| C4 | 冻结 μ | 信任是否真参与策略 |
| C5 | 固定 θ=θ_base,仅 λ 变 | 仅配给互惠能否解释 B4(反证) |
| C6 | 信任更新但不进观测 | 策略是否真用关系信息 |
| **C7** | **η=0**(关互惠效用) | **改善是否非奖励塑形所致** |

- **最关键的判读**:
  - **C7(η=0)仍优于 B1/B2 且伴随紧张期前 G_t 提升** → 互惠有**结构性**协调价值、η 仅放大 → **强结论**,
    评审无法以"你用奖励造互惠"反驳。
  - C7 ≈ B1/B2、仅 η>0 时 B4 取胜 → 价值由关系动机**使能**,作诚实弱表述(因评价指标独立于奖励,仍非循环)。
  - B4 明显优于 C5 且 SR 下降伴随 G_t 提升 → 改善走**产能路径**而非零和搬运。
  - **关系专属性**由 C3(去专属权重)+ C6(去关系观测)联合识别(C7 单独只反证非奖励塑形)。

---

## 6. 实验四/五/六(稳健性 / 边界 / 外部效度)

这三个在 `model/0626.md` §4.6.4–4.6.6 已完整设计,但 **`run_sweep` 暂未内置一键预设**,目前通过覆盖
环境参数或单独脚本运行(后续可往 `run_sweep.EXPERIMENTS` 补条目)。

- **实验四(参数敏感性)**:对 B4(及对照)扫 `K∈{1.0,1.2,1.5}mD̄`、`ρ_max∈{0.02,0.05,∞}`、
  `W_on∈{18,26,34}`、`λ_max`、`α/δ`、`σ_D`、`c_rep`、`η/λ_S/λ_J`,逐维固定其余。重点验:`ρ_max→∞`
  时跨期协调价值消失(惯性是机制前提);`η` 小时改善仍可由独立指标观测(与 C7 互补)。
- **实验五(规模/范围/竞争)**:`m∈{2,3,5,7,9}`(单电企需求随 m 调整保压力比)、互惠范围
  `h∈{1,…,⌊m/2⌋}`(煤企互惠项已按 `1/h` 平均,分离范围与奖励尺度)、需求相关性
  `ρ∈{0,…,1}`(`demand_corr` / `demand_common_pool`)。
- **实验六(真实数据回放)**:冻结 B1/B2/B3/B4 策略,在真实电力负荷 + 三档压力产能 `K_stress`
  上平行回放,区分"学习型(B2)"与"动态互惠(B4)"改进。

> 覆盖单个参数示例(单组单种子):
> ```bash
> python cleanmarl/run_sweep.py --config configs/happo_cpdre.yaml --group B4 --seeds 42 \
>   --env-override rho_max=0.02   # 若 run_sweep 暂不支持 --env-override,改对应 yaml 后再跑
> ```

---

## 7. 组别 → 环境机制映射(`run_sweep.GROUP_ENV_MAP`)

| 组 | mechanism | allocation | 算法 | 备注 |
| :-- | :-- | :-- | :-- | :-- |
| A1/A2 | none | fair | rule_a1/a2 | 规则基线,θ 固定季节排程 |
| A3/A4/A5/A6 | b2 | fair | ippo/mappo/happo(A6 低扰动) | 学习无互惠,煤企学 θ |
| B1 | none | fair | rule_b1 | 公平规则基准 |
| B2 | b2 | weighted | (YAML) | 非关系动态协调基准,λ=1,无记忆,无 ημg |
| B3 | b3 | weighted | (YAML) | 规则互惠:学 θ + λ 依 χ̂ 触发,无记忆 |
| B4 | b4 | weighted | (YAML) | 完整关系专属互惠,学 θ 和 λ,**含 ημg** |
| C1–C6 | b4 变体 | weighted | (YAML) | C2=disable_g_u, C3=fixed_lambda, C4=freeze_memory, C5=freeze_theta, C6=去关系观测 |
| C7 | b4 | weighted | (YAML) | η=0(`use_reciprocity_reward=False`),去互惠效用反证 |

`--group` 会自动绑定机制,避免"策略 B1 但环境 B4"错配;规则组用 `force_algo` 强制规则策略,所以
`--config` 用哪个 yaml 都行(只取它的环境 + 超参)。单组跑:`--group B4`;单种子:`--seeds 42`;
冒烟:`--timesteps 1560`(1 rollout,**只验管线,数值无意义**)。

---

## 8. 为什么能做配对检验(固定评估轨迹)

训练随机性与评估随机性**故意分开**:

- 训练种子(`--seeds 40..44`)→ 决定训练随机流(5 个种子 = 方法重复 5 次)。
- 评估走 `env.eval_reset(episode_idx)`,内部固定 `rng = default_rng(9000 + episode_idx)`,**与训练种子无关**。
  所以任何组别/种子,评估都在**同一批 64 条固定轨迹**(episode 0..63 ↔ 种子 9000..9063)上跑。

→ `B4-seed40-ep5` 和 `B2-seed40-ep5` 面对**完全相同的需求实现**,差异只来自机制本身。这就是
`analyze_experiments.py` 按 (seed, episode) 配对做 Wilcoxon 的前提,消掉"剧本难易"这一最大噪声源。

---

## 9. 分析脚本

```bash
python cleanmarl/analyze_experiments.py --experiment exp2          # 出组间表 + 默认配对检验
python cleanmarl/analyze_experiments.py --experiment exp2 --groups B2,B3,B4 --pairs B4:B2 \
       --metric system_profit,SR --csv out.csv
```

- 默认配对:exp2 = `B4:B2 / B4:B3 / B4:B1`;exp3 = `C1:C7 / C1:C5 / C1:C2/C3/C4/C6`。
- 报告指标均为**奖励加项之外的独立量**:SR、SR_1、SR_N、Jain、系统/煤企/电企利润、θ 路径、爬坡命中率、
  紧张期前 Ḡ、λ₁、g^U/g^C、μ 轨迹。
- 每个 run 目录含 `config.json / progress.csv / episode_summary.csv / run_meta.json`。

---

## 10. 快速冒烟

```bash
# 单组管线冒烟(1 rollout, cpu): 确认端到端不崩
python cleanmarl/run_sweep.py --config configs/happo_cpdre.yaml --group B4 --seeds 42 --timesteps 1560 --device cpu
```

最近一次(0626 落地后)机制冒烟确认:平台季节(旺 1.2 / 淡 0.8 / 年均 1.0)、价格恒 1、χ 两边穿越
(宽松≈48% / 紧张≈52%)、g^U/g^C 真实点火(B4 日志 mean_g_u≈0.42、mean_g_c≈0.10、μ_c→0.74、
λ₁≈1.48)、b2 干净(Σ|g|=0、λ≡1)。**正式结论仍需跑满 `total_timesteps`(1e5)× 5 种子。**

---

## 11. 代码结构

```text
cleanmarl/
├── algos/{happo,mappo,ippo,rule_policy}.py   # 算法 + 规则/固定基线
├── configs/{default,happo_cpdre,ippo_cpdre,mappo_cpdre}.yaml
├── core/{trainer,logger,episode_logger}.py   # 训练循环 + 指标/进度日志
├── envs/cpdre_wrapper.py                      # CPDRE numpy wrapper
├── models/models.py                           # GRUActor / CentralizedCritic
├── train.py            # 单次训练入口(YAML)
├── run_sweep.py        # 多种子/多组别扫描 + --experiment 预设 + GROUP_ENV_MAP
└── analyze_experiments.py  # 组间表 + 配对 Wilcoxon/bootstrap
```

环境本体:`custom_envs/coal_power_direct_reciprocity_env.py`。

---

## 12. 0626 对齐状态与已知缺口

**已对齐 0626**:平台型季节(均值守恒)、价格固定 1、`g^C` 基准产能公平基准 + `g^U` 剩余供给减少、
互惠效用 ημg 进主模型(C7 可关)、`K=1.2mD̄` 留旺季余量、`γ=0.99`、`c_rep=1.70`、连续动作 squash 后算 log-prob、
B5 已删 / B3→b3 / 新增 C7。

**已知缺口**(详见 `model/0625_issues.md`、`model/cleanmarl_0611_issues.md`):

- 实验四/五/六未在 `run_sweep` 做一键预设(目前靠改 yaml / 单独脚本)。
- HAPPO advantage 迭代是相邻 agent 简化传播,非完整 Kuba 2022 联合 ratio(已声明不依赖单调性)。
- 中途评估从不触发(`eval_freq` 与 rollout 步数对不上),只有训练结束的 final eval 写 `phase=eval`;
  中途 Ctrl+C 的 run 无 eval 行。
- 多互惠 `h>1`、多电企 `m>3` 的观测/日志仍有 `u1/u2/u3`、`pad3` 等固定写法,扩展前需先改 schema。
