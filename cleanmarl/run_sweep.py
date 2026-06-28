"""
多种子扫描运行器 (model 4.3 table: 训练种子 {40,41,42,43,44}).

所有训练种子共享同一批 64 条固定评估轨迹 (eval_seed_offset=9000), 因此不同
组别、不同种子的评估结果可做配对比较 (Wilcoxon / paired bootstrap).

--group 参数把实验组别自动绑定到对应环境机制, 避免手改 YAML 出现错配
(例如 "策略 B1 但环境 B4"). 组别到环境的映射见 GROUP_ENV_MAP.

用法:
    # 学习型组别 (algo 由 YAML 决定)
    python run_sweep.py --config configs/happo_cpdre.yaml --group B4 --seeds 40,41,42,43,44
    python run_sweep.py --config configs/happo_cpdre.yaml --group B5 --seeds 40,41,42,43,44
    # 规则/固定基线组别 (强制 algo=rule_xxx, 忽略 YAML 的 algo)
    python run_sweep.py --config configs/happo_cpdre.yaml --group B1 --seeds 42
"""
import sys
import argparse
import copy
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from cleanmarl.train import load_config, merge_config, build_policy
from cleanmarl.envs.cpdre_wrapper import make_cpdre_env


# 实验组别 -> 环境机制 + 策略类型 (model 0611(1).md).
# mechanism_mode: none/b2/b4/b5/dynamic/long_contract/trigger
# allocation_mode: fair/weighted (b4/dynamic 需 weighted 才能让 lambda 生效)
# freeze_theta: C5 ablation (固定产能路径, 只学 lambda)
# force_algo: None=沿用YAML算法; "rule_xxx"/"fixed"=强制规则策略
GROUP_ENV_MAP = {
    # 实验一: 基线 + 算法选择 (均无互惠).
    # A1/A2 规则基线: b2 模式 (煤企用动作里的派生 theta, lambda 强制 1, 公平配给),
    #   策略为 RulePolicy 派生式 base-stock (env.baseline_coal_theta/power_omega).
    #   *必须用 b2 不用 none*: none 会把 theta 钉死成手设季节排程, 且历史上 none 还
    #   误用遗留两档需求(1.0/1.4)而 b 模式用平台需求(0.8/1.2), 两者不可比 (已修).
    #   用 b2 让基线与学习组(A3-A6/B2-B4)面对完全相同的需求与执行管线.
    # A3/A4/A5/A6 学习基线: b2 模式让煤企从公共状态学 theta (doc 4.5.3).
    "A1": dict(mechanism_mode="b2",      allocation_mode="fair",     force_algo="rule_a1"),
    "A2": dict(mechanism_mode="b2",      allocation_mode="fair",     force_algo="rule_a2"),
    "A3": dict(mechanism_mode="b2",      allocation_mode="fair",     force_algo="ippo"),
    "A4": dict(mechanism_mode="b2",      allocation_mode="fair",     force_algo="mappo"),
    "A5": dict(mechanism_mode="b2",      allocation_mode="fair",     force_algo="happo"),
    "A6": dict(mechanism_mode="b2",      allocation_mode="fair",     force_algo="happo", low_noise=True),
    # 实验二: 机制比较 (核心)
    "B1": dict(mechanism_mode="b2",      allocation_mode="fair",     force_algo="rule_b1"),
    "B2": dict(mechanism_mode="b2",      allocation_mode="weighted", force_algo=None),
    "B3": dict(mechanism_mode="b3",      allocation_mode="weighted", force_algo=None),
    "B4": dict(mechanism_mode="b4",      allocation_mode="weighted", force_algo=None),
    # B5 双边静态长协 (对照真实电煤中长期合同): 学 theta; 淡季 U1 提高 omega 预先囤煤
    #   (买方承购义务), 旺季煤企对 U1 静态高保供权重 w_high (卖方保供义务); 无互惠
    #   奖励/记忆/观测 -> 静态双边合同, 不随关系记忆动态调节. 对照 B3(按压力触发)与
    #   B4(按关系记忆学习), 隔离"动态关系互惠 vs 静态双边长协".
    "B5": dict(mechanism_mode="long_contract", allocation_mode="weighted", force_algo=None,
               no_recip_reward=True, no_recip_in_obs=True),
    # 实验三: 消融 (均为 b4 变体)
    "C1": dict(mechanism_mode="b4",      allocation_mode="weighted", force_algo=None),
    "C2": dict(mechanism_mode="b4",      allocation_mode="weighted", force_algo=None, disable_g_u=True),
    "C3": dict(mechanism_mode="b4",      allocation_mode="weighted", force_algo=None, fixed_lambda=1.0),
    "C4": dict(mechanism_mode="b4",      allocation_mode="weighted", force_algo=None, freeze_memory=True),
    "C5": dict(mechanism_mode="b4",      allocation_mode="weighted", force_algo=None, freeze_theta=True),
    "C6": dict(mechanism_mode="b4",      allocation_mode="weighted", force_algo=None, no_recip_in_obs=True),
    "C7": dict(mechanism_mode="b4",      allocation_mode="weighted", force_algo=None, no_recip_reward=True),
}

# 实验 -> 组别列表 (一条命令跑完整个实验).
#   exp1: 基线 + 算法选择 (无互惠)
#   exp2: 直接互惠机制主实验 (B4 vs B2 为头条对比; 见 model 0626 式4-27)
#   exp3: 机制来源消融 (均为 b4 变体; C7=去互惠效用 eta=0 反证)
EXPERIMENTS = {
    "exp1": ["A1", "A2", "A3", "A4", "A5", "A6"],
    "exp2": ["B1", "B2", "B3", "B4"],
    "exp3": ["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
}


def apply_group(cfg, group):
    """把组别映射应用到 cfg (env + experiment.algo), 返回新 cfg (深拷贝)."""
    if group is None:
        return cfg
    key = str(group).upper()
    if key not in GROUP_ENV_MAP:
        raise ValueError(f"未知实验组别: {group}. 可选: {sorted(GROUP_ENV_MAP.keys())}")
    spec = GROUP_ENV_MAP[key]
    cfg = copy.deepcopy(cfg)
    cfg['env']['mechanism_mode'] = spec['mechanism_mode']
    cfg['env']['allocation_mode'] = spec['allocation_mode']
    if spec.get('freeze_theta'):
        cfg['env']['freeze_theta'] = True
    if spec.get('disable_g_u'):
        cfg['env']['disable_g_u'] = True
    if spec.get('freeze_memory'):
        cfg['env']['freeze_memory'] = True
    if spec.get('no_recip_in_obs'):
        cfg['env']['include_reciprocity_in_obs'] = False
    if spec.get('no_recip_reward'):
        # C7 ablation: turn off the eta*mu*g reciprocity utility (model 0626 4.6.3).
        cfg['env']['use_reciprocity_reward'] = False
    if 'fixed_lambda' in spec:
        cfg['env']['fixed_coal_weight'] = float(spec['fixed_lambda'])
    if spec.get('low_noise'):
        cfg['env']['demand_mode'] = 'low_noise'
    if spec.get('force_algo'):
        cfg['experiment']['algo'] = spec['force_algo']
    # 记录组别, run_one 用它生成干净的 run 目录名 + run_meta.json.
    cfg['experiment']['group'] = key
    return cfg


def run_one(cfg, seed, device=None, timesteps=None):
    """单次训练/评估运行, 返回 log 目录名."""
    train_config = {
        'episode_len': cfg['train']['episode_len'],
        'rollout_length': cfg['train']['rollout_length'],
        'num_sgd_iter': cfg['train']['num_sgd_iter'],
        'log_freq': cfg['train']['log_freq'],
        'chunk_len': cfg['train'].get('chunk_len', 0),
        'gamma': cfg['ppo']['gamma'],
        'gae_lambda': cfg['ppo']['gae_lambda'],
        'clip_param': cfg['ppo']['clip_param'],
        'value_clip_param': cfg['ppo']['value_clip_param'],
        'entropy_coef': cfg['ppo']['entropy_coef'],
        'value_loss_coef': cfg['ppo']['value_loss_coef'],
        'max_grad_norm': cfg['ppo']['max_grad_norm'],
        'actor_lr': cfg['lr']['actor_lr'],
        'critic_lr': cfg['lr']['critic_lr'],
        'hidden_dim': cfg['model']['hidden_dim'],
        'rnn_layers': cfg['model'].get('rnn_layers', 1),
        'device': cfg['system']['device'],
        'log_dir': cfg['system']['log_dir'],
        'total_timesteps': cfg['train']['total_timesteps'],
        'log_step_details': cfg['train'].get('log_step_details', False),
        'eval_episodes': cfg['train'].get('eval_episodes', 64),
        'eval_seed_offset': cfg['env'].get('eval_seed_offset', 9000),
    }

    cfg['env']['seed'] = seed
    algo = cfg['experiment']['algo']
    group = cfg['experiment'].get('group')
    # 干净的 run 目录名: {GROUP}_{algo}_seed{seed} (便于 analyze_experiments 解析).
    if group:
        exp_name = f"{group}_{algo}_seed{seed}"
    else:
        exp_name = f"{cfg['experiment'].get('name', algo)}_seed{seed}"
    train_config['experiment_name'] = exp_name
    if device:
        train_config['device'] = device
    if timesteps:
        train_config['total_timesteps'] = timesteps
    if train_config['device'] == 'cuda' and not torch.cuda.is_available():
        train_config['device'] = 'cpu'

    # 写 run_meta.json: 让分析脚本稳健地把 run 目录映射到 (group, algo, seed, 机制).
    run_dir = Path(train_config['log_dir']) / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "group": group,
        "algo": algo,
        "seed": seed,
        "experiment_name": exp_name,
        "mechanism_mode": cfg['env']['mechanism_mode'],
        "allocation_mode": cfg['env']['allocation_mode'],
        "demand_mode": cfg['env']['demand_mode'],
        "total_timesteps": train_config['total_timesteps'],
        "eval_episodes": train_config['eval_episodes'],
    }
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 固定全局随机种子 (torch + numpy): 否则网络初始化/动作采样每个 run 都不同,
    # 制造 ±13/种子 的 run 间噪声底, 把 ~6 的机制效应 (B4 vs B2) 淹没. 固定后同一
    # seed 下不同机制组共享相同初始化 -> 机制差异被干净隔离, 统计功效大增.
    from cleanmarl.utils import set_seed
    set_seed(int(seed))

    env = make_cpdre_env(cfg['env'])
    policy = build_policy(algo, env, train_config)
    try:
        policy.train(total_timesteps=train_config['total_timesteps'],
                     eval_freq=cfg['train']['eval_freq'])
    except KeyboardInterrupt:
        print(f"\n⚠️ seed {seed} 中断")
    finally:
        save_path = f"{cfg['system']['checkpoint_dir']}/{train_config['experiment_name']}_final.pt"
        try:
            policy.save_checkpoint(save_path)
        except Exception as e:
            print(f"checkpoint 保存失败: {e}")
        env.close()
    return train_config['experiment_name']


def run_group(base_cfg, group, seeds, device=None, timesteps=None):
    """对单个组别跑所有 seed."""
    cfg = apply_group(base_cfg, group)
    print("=" * 80)
    print(f"Group {group}: algo={cfg['experiment']['algo']} "
          f"seeds={seeds} eval_episodes={cfg['train'].get('eval_episodes', 64)}")
    print(f"  env: mechanism={cfg['env']['mechanism_mode']} "
          f"allocation={cfg['env']['allocation_mode']} "
          f"demand={cfg['env']['demand_mode']}")
    print("=" * 80)
    for seed in seeds:
        print(f"\n---------- {group} seed {seed} ----------")
        run_one(cfg, seed, device=device, timesteps=timesteps)


def main():
    parser = argparse.ArgumentParser(description="多种子扫描 (--group/--experiment 自动绑定环境机制)")
    parser.add_argument("--config", type=str, default="configs/happo_cpdre.yaml")
    parser.add_argument("--group", type=str, default=None,
                        help=f"单个实验组别, 自动绑定 mechanism_mode/allocation_mode. "
                             f"可选: {sorted(GROUP_ENV_MAP.keys())}")
    parser.add_argument("--experiment", type=str, default=None,
                        help=f"跑完整实验 (展开为多个组别). 可选: {sorted(EXPERIMENTS.keys())}")
    parser.add_argument("--seeds", type=str, default="40,41,42,43,44",
                        help="训练种子, 逗号分隔")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()

    config_path = project_root / "cleanmarl" / args.config
    base_cfg = load_config(config_path)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.experiment:
        key = args.experiment.lower()
        if key not in EXPERIMENTS:
            raise ValueError(f"未知实验: {args.experiment}. 可选: {sorted(EXPERIMENTS.keys())}")
        groups = EXPERIMENTS[key]
        print(f"\n████ 实验 {key}: 组别 {groups} × seeds {seeds} ████\n")
        for group in groups:
            run_group(base_cfg, group, seeds, device=args.device, timesteps=args.timesteps)
        print(f"\n✅ 实验 {key} 完成 ({len(groups)} 组 × {len(seeds)} 种子)")
    else:
        run_group(base_cfg, args.group, seeds, device=args.device, timesteps=args.timesteps)
        print("\n✅ Sweep 完成")


if __name__ == "__main__":
    main()
