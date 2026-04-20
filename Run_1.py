import sys
import os

# 自动获取当前文件的目录，然后找到项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 假设脚本在项目子目录中
sys.path.insert(0, project_root)
from SF_TRON_FP.SRC.Env.TronEnv import TronEnv
from SF_TRON_FP.SRC.PPO.Actor_Critic import Actor_Critic
from SF_TRON_FP.SRC.Config.Config import *

maximum_step = PPOCfg.PPOParam.maximum_step
episode = PPOCfg.PPOParam.episode
time_per_epi = EnvCfg.EnvParam.dt * maximum_step
train = EnvCfg.EnvParam.train
PPO_1 = Actor_Critic(PPOCfg, EnvCfg)
if not train:
    PPO_1.load_best_model()

Env = TronEnv(EnvCfg, RobotCfg, PPOCfg)
import torch

Env.prim_initialization(reset_all=True)
for epi in range(episode):
    print(f"===================episode: {epi}===================")
    if epi % int(5 / time_per_epi + 1) == 0:
        Env.resample_command()
        Env.apply_disturbance()
    for step in range(maximum_step):
        """获取当前状态"""
        state = Env.get_current_observations()
        state[:, 33:] = 0  # basic state 之后就是地图信息，第一阶段机器人盲走

        """做动作"""
        action, scaled_action = PPO_1.sample_action(state, deterministic=not train)

        """更新环境"""
        Env.update_world(scaled_action=scaled_action)

        """获取下一个状态"""

        next_state = Env.get_next_observations()
        next_state[:, 33:] = 0

        """计算奖励 判断是否结束"""

        reward, over, truncated = Env.compute_reward()

        """存储经验"""
        if train:
            PPO_1.store_experience(state,
                                   action,
                                   next_state,
                                   reward,
                                   over,
                                   step)

        """重置挂掉的机器人"""
        over += truncated
        Env.prim_initialization(torch.nonzero(over.flatten()).flatten())

    """每个回合结束后训练一次"""
    if train:
        PPO_1.update()
        Env.print_reward_sum()
