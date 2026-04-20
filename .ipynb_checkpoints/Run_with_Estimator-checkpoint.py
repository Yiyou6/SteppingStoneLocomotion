import sys
import os

# 自动获取当前文件的目录，然后找到项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 假设脚本在项目子目录中
sys.path.insert(0, project_root)
from SF_TRON_FP.SRC.Env.TronEnv import TronEnv
from SF_TRON_FP.SRC.PPO.Actor_Critic import Actor_Critic
from SF_TRON_FP.SRC.Config.TS_Config import *
from SF_TRON_FP.SRC.Estimator.Estimator import *
from SF_TRON_FP.SRC.Plotter.ImagePlotter import *

Img = ImagePlotter(image_number=2)
maximum_step = PPOCfg.PPOParam.maximum_step
episode = PPOCfg.PPOParam.episode
time_per_epi = EnvCfg.EnvParam.dt * maximum_step
train = EnvCfg.EnvParam.train
PPO_3 = Actor_Critic(PPOCfg, EnvCfg, index=2)
Estimator_1 = Estimator(PPOCfg, EnvCfg, index=1)
if not train:
    PPO_3.load_best_model()
    Estimator_1.load_each_epi_model()

Env = TronEnv(EnvCfg, RobotCfg, PPOCfg)
import torch

Env.prim_initialization(reset_all=True)
for epi in range(episode):
    print(f"===================episode: {epi}===================")
    if epi % int(5 / time_per_epi + 1) == 0:
        Env.resample_command()
    if epi % int(2 / time_per_epi + 1) == 0:
        Env.apply_disturbance()
    state = Env.get_current_observations()
    Estimator_1.store_forward_state(state[:, :33])
    for step in range(maximum_step):
        """获取当前状态"""
        state = Env.get_current_observations()
        state[:, 33:] = 0  # basic state 之后就是地图信息，第一阶段机器人盲走
        estimated_privilege_state = Estimator_1.get_estimate_output()
        full_state = torch.concatenate((state,estimated_privilege_state),dim=-1)
        if not train:
            privilege_state = Env.get_privilege()
            est = Estimator_1.get_estimate_output()

            Img.append(epi * maximum_step + step, 100 * est[:,5:6][0, 0].item(), 0)
            Img.append(epi * maximum_step + step, 100 * privilege_state[:, 5:6][0, 0].item(), 1)
            Img.animation_plot()

        """做动作"""
        action, scaled_action = PPO_3.sample_action(full_state, deterministic=not train)

        """更新环境"""
        Env.update_world(scaled_action=scaled_action)

        """获取下一个状态"""

        next_state = Env.get_next_observations()
        next_state[:, 33:] = 0
        next_privilege_state = Env.get_privilege()
        Estimator_1.store_forward_state(next_state[:, :33])
        next_estimated_privilege_state = Estimator_1.get_estimate_output()
        next_full_state = torch.concatenate((next_state, next_estimated_privilege_state), dim=-1)

        """计算奖励 判断是否结束"""

        reward, over, truncated = Env.compute_reward()

        """存储经验"""
        if train:
            PPO_3.store_experience(full_state,
                                   action,
                                   next_full_state,
                                   reward,
                                   over,
                                   step)

            Estimator_1.store_new_state_and_output(next_state[:, :33],
                                                   next_privilege_state,
                                                   step,
                                                   over)

        """重置挂掉的机器人"""
        over += truncated
        Env.prim_initialization(torch.nonzero(over.flatten()).flatten())

    """每个回合结束后训练一次"""
    if train:
        PPO_3.update()
        Estimator_1.update()
        Env.print_reward_sum()
