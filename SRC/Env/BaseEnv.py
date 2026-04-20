from SF_TRON_FP.SRC.Utils.Transformation import *


class BaseEnv:
    def __init__(self, EnvCfg, RobotCfg, PPOCfg):
        """
        用于创建基础的环境所需的参数和函数，省的重复写
        :param EnvCfg:
        :param RobotCfg:
        :param PPOCfg:

        通用：即任何RL任务都需要的
        Task Dependent：即根据不同的任务需要加入的参数
        """

        """通用：初始化环境基本参数"""
        self.device = EnvCfg.EnvParam.device
        self.dt = EnvCfg.EnvParam.dt
        self.sub_step = EnvCfg.EnvParam.sub_step
        self.train = EnvCfg.EnvParam.train
        self.agents_num = (EnvCfg.EnvParam.agents_num - EnvCfg.EnvParam.agents_num_in_play) * self.train + \
                          EnvCfg.EnvParam.agents_num_in_play
        self.all_agent_indices = torch.arange(self.agents_num, device=self.device)
        import time as t
        self.t_module = t
        self.start_time = t.time()
        self.time = torch.zeros((self.agents_num, 1), device=self.device)

        """通用：初始化action相关"""
        self.actuator_num = RobotCfg.ActuatorParam.actuator_num
        self.action = torch.zeros((self.agents_num, self.actuator_num), device=self.device)  # 动作
        self.prev_action = torch.zeros((self.agents_num, self.actuator_num), device=self.device)  # 上一次动作

        """Task Dependent：初始化环境变量"""
        self.file_path = EnvCfg.EnvParam.file_path  # 因为isaaclab需要，用于指定机器人模型文件的路径
        self.friction_coef = EnvCfg.EnvParam.friction_coef  # 因为isaaclab需要这个来指定地面摩擦力
        self.backend = EnvCfg.EnvParam.backend  # 因为isaaclab需要这个来指定isaaclab用cpu计算还是gpu
        self.headless = EnvCfg.EnvParam.headless  # 因为isaaclab需要，这个参数指定是否带UI

        """Task Dependent： 初始化域随机化参数"""
        self.DomainRandomizationCfg = RobotCfg.DomainRandomizationCfg
        self.Kp = FT([RobotCfg.ActuatorParam.Kp] * self.agents_num)
        self.Kd = FT([RobotCfg.ActuatorParam.Kd] * self.agents_num)
        self.default_PD_angle = FT([RobotCfg.ActuatorParam.default_PD_angle] * self.agents_num)
        Kp_range = self.DomainRandomizationCfg.Kp_range
        Kd_range = self.DomainRandomizationCfg.Kd_range
        self.Kp = self.Kp * (1 + Kp_range * rand_num_like(self.Kp))
        self.Kd = self.Kd * (1 + Kd_range * rand_num_like(self.Kd))
        self.action_delay_range = self.DomainRandomizationCfg.action_delay_range
        self.external_body_force_range = self.DomainRandomizationCfg.external_body_force_range

        """Task Dependent：初始化机器人出生状态的范围"""
        self.initial_body_linear_vel_range = RobotCfg.InitialState.initial_body_linear_vel_range
        self.initial_body_angular_vel_range = RobotCfg.InitialState.initial_body_angular_vel_range
        self.initial_joint_pos_range = RobotCfg.InitialState.initial_joint_pos_range
        self.initial_joint_vel_range = RobotCfg.InitialState.initial_joint_vel_range
        self.initial_height = RobotCfg.InitialState.initial_height
        self.initial_euler_angle_range = RobotCfg.InitialState.initial_euler_angle_range

        """Task Dependent：初始化额外机器人参数"""
        self.vel_cmd = torch.zeros((self.agents_num, 1), device=self.device)  # 速度指令,0表示暂停，1表示前进
        self.target_ori = torch.zeros((self.agents_num, 3), device=self.device)
        self.phase = torch.zeros((self.agents_num, 1), device=self.device)
        self.L_feet_air_time = torch.zeros((self.agents_num, 1), device=self.device)  # 左脚离地时间
        self.R_feet_air_time = torch.zeros((self.agents_num, 1), device=self.device)  # 右脚离地时间

        self.action_history = torch.zeros((self.agents_num, self.sub_step, self.actuator_num),
                                          device=self.device)  # 动作历史
        self.action_delay_idx = torch.randint(0, self.action_delay_range, (self.agents_num,),
                                              device=self.device)  # 延迟多少步
        self.external_body_force = torch.zeros((self.agents_num, 3),
                                               device=self.device)  # the dim 1 is necessary for isaac lab
        self.external_body_torques = torch.zeros((self.agents_num, 3), device=self.device)

        """Task Dependent： 初始化奖励和"""
        self.max_step = PPOCfg.PPOParam.maximum_step
        self.vel_tracking_reward_sum = 0
        self.body_height_tracking_reward_sum = 0
        self.body_ori_tracking_reward_sum = 0
        self.foot_constraint_reward_sum = 0
        self.stand_still_reward_sum = 0
        self.single_support_reward_sum = 0
        self.foot_air_time_reward_sum = 0
        self.Termination_reward_sum = 0

        """Task Dependent： 导入Isaac Sim 库"""
        from ..Env.SoftwareSetup import App_Setup
        App_Setup(self.device, self.headless)
        from ..Env.SceneSetup import create_environment

        """Task Dependent：初始化Isaac Sim环境"""
        self.sim, self.scene = create_environment(self.file_path,
                                                  self.dt,
                                                  self.sub_step,
                                                  self.agents_num,
                                                  self.device,
                                                  self.DomainRandomizationCfg)

        """通用：指定需要0初始化的参数列表，在prim_initialization中会被重置为0"""
        self.reset_list = [self.time,
                           self.phase,
                           self.L_feet_air_time,
                           self.R_feet_air_time,
                           self.prev_action,
                           self.action,
                           self.action_history]

    def prim_initialization(self, agent_index=None, reset_all=False):
        """
        通用：用于重置机器人状态的函数，但是具体的重置内容是根据任务需要来定的
        :param reset_all:
        :param agent_index:  哪个序号的机器人挂了
        :return:
        重置指定序号机器人的状态

        该函数在训练开始前（进入第一个episode前）和机器人挂掉时调用
        """

        if reset_all:
            agent_index = torch.arange(self.agents_num, device=self.device)

        num_agents = len(agent_index)
        if num_agents == 0:
            return

        # 生成随机初始化数据
        initial_linear_vel = self.initial_body_linear_vel_range * rand_num((num_agents, 3), device=self.device)
        initial_linear_vel[:, -1] = 0  # no push in z

        initial_angular_vel = self.initial_body_angular_vel_range * rand_num((num_agents, 3), device=self.device)

        initial_joint_pos = self.initial_joint_pos_range * rand_num((num_agents, 8), device=self.device)
        initial_joint_vel = self.initial_joint_vel_range * rand_num((num_agents, 8), device=self.device)
        initial_body_v_w = torch.concatenate((initial_linear_vel, initial_angular_vel), dim=1)

        initial_roll = self.initial_euler_angle_range[0] * rand_num((num_agents, 1), device=self.device)
        initial_pitch = self.initial_euler_angle_range[1] * rand_num((num_agents, 1), device=self.device)
        initial_yaw = self.initial_euler_angle_range[2] * rand_num((num_agents, 1), device=self.device)
        initial_euler_angle = torch.concatenate((initial_roll, initial_pitch, initial_yaw), dim=-1)
        initial_quat = euler_to_quaternion(initial_euler_angle)

        # 设置速度命令和时间
        """重新初始化额外机器人参数"""
        for i in range(len(self.reset_list)):
            self.reset_list[i][agent_index] = 0

        # 获取prim并设置身体速度
        self.scene["robot"].reset(env_ids=agent_index.cpu().tolist())
        root_state = self.scene["robot"].data.default_root_state[agent_index].clone()
        root_state[:, :3] += self.scene.env_origins[agent_index]
        root_state[:, 2] += self.initial_height
        root_state[:, 3:7] = initial_quat
        self.scene["robot"].write_root_pose_to_sim(root_state[:, :7], env_ids=agent_index)
        self.scene["robot"].write_root_velocity_to_sim(root_velocity=initial_body_v_w, env_ids=agent_index)
        self.scene["robot"].write_joint_state_to_sim(position=initial_joint_pos,
                                                     velocity=initial_joint_vel,
                                                     env_ids=agent_index)
        self.scene.write_data_to_sim()
        self.scene.update(dt=0)

    def resample_command(self,):  # Only activate in walking, not stepping stone
        """
        Task Dependent: 用于模拟高层控制器的命令更新，给每个机器人生成一个新的速度指令，70%概率前进，30%概率原地不动
        :return:
        """

        self.vel_cmd = torch.rand((self.agents_num, 1), device=self.device)
        self.vel_cmd = (self.vel_cmd > 0.3).float()  # 70%概率前进，30%概率原地不动

    def apply_disturbance(self):
        """
        Task Dependent: 用于模拟外力扰动，给20%的机器人加一个随机的外力
        :return:
        """
        is_apply = torch.rand((self.agents_num, 1), device=self.device) > 0.8  # 给20%的人加外力
        self.external_body_force = rand_num((self.agents_num, 3), self.device) * is_apply.float()
        self.external_body_torques = rand_num((self.agents_num, 3), self.device) * is_apply.float()

        self.external_body_force[:, 0] *= self.external_body_force_range[0]
        self.external_body_force[:, 1] *= self.external_body_force_range[1]
        self.external_body_force[:, 2] *= self.external_body_force_range[2]
        external_body_force = rand_num((self.agents_num, 1, 3), self.device)
        external_body_torques = rand_num((self.agents_num, 1, 3), self.device) * 0

        external_body_force[:, 0, :] = self.external_body_force

        self.scene["robot"].set_external_force_and_torque(external_body_force,
                                                          external_body_torques,
                                                          body_ids=[0],
                                                          is_global=True)

    def append_action_history(self, action: torch.Tensor):
        """
        Task Dependent: 用于模拟延迟的： u（t - delay），把历史的动作命令记下，这样选择动作的时候可以选择历史动作
        :param action:
        :return:
        """
        self.action_history[:, 1:, :] = self.action_history[:, :-1, :].clone()
        self.action_history[:, 0, :] = action.clone()
