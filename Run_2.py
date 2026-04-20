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

PPO_1 = Actor_Critic(PPOCfg, EnvCfg, index=0)
PPO_1.load_best_model()
PPO_2 = Actor_Critic(PPOCfg, EnvCfg, index=1)
if not train:
    PPO_2.load_best_model()
env = TronEnv(EnvCfg, RobotCfg, PPOCfg)
import torch

env.prim_initialization(reset_all=True)
for epi in range(episode):
    print(f"===================episode: {epi}===================")
    if epi % int(5 / time_per_epi + 1) == 0:
        env.resample_command()
        env.apply_disturbance()
    for step in range(maximum_step):
        """获取当前状态"""
        state = env.get_current_observations()

        state_no_camera = state.clone()
        state_no_camera[:, 33:] = 0

        """做动作"""

        action1, scaled_action1 = PPO_1.sample_action(state_no_camera, deterministic=True)
        action2, scaled_action2 = PPO_2.sample_action(state, deterministic=not train)

        """更新环境"""
        env.update_world(scaled_action=scaled_action1 * 0.2 + scaled_action2 * 0.8)

        """获取下一个状态"""

        next_state = env.get_next_observations()

        """计算奖励 判断是否结束"""

        reward, over, truncated = env.compute_reward()

        """存储经验"""
        if train:
            PPO_2.store_experience(state,
                                   action2,
                                   next_state,
                                   reward,
                                   over,
                                   step)

        """重置挂掉的机器人"""
        over += truncated
        env.prim_initialization(torch.nonzero(over.flatten()).flatten())

    """每个回合结束后训练一次"""
    if train:
        PPO_2.update()
        env.print_reward_sum()
