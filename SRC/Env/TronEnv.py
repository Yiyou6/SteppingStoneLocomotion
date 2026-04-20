from SF_TRON_FP.SRC.Env.BaseEnv import *
class TronEnv(BaseEnv):
    def __init__(self, EnvCfg, RobotCfg, PPOCfg):
        super().__init__(EnvCfg, RobotCfg, PPOCfg)

    """-------------------以上均为初始化代码-----------------------"""
    """-------------------以上均为初始化代码-----------------------"""
    """-------------------以上均为初始化代码-----------------------"""

    """-------------------以下均为环境运行代码-----------------------"""
    """-------------------以下均为环境运行代码-----------------------"""
    """-------------------以下均为环境运行代码-----------------------"""

    """机器人状态更新"""

    def get_current_observations(self):
        """
        通用：获取机器人当前状态的函数，包含了所有RL任务都需要的状态信息； 记为St。返回什么是Task Dependent
        :return:
        """
        # 获取机器人
        # #——————————————————————获取当前时刻状态————————————————————————————————##
        self.body_pos, self.body_ori = self.scene["imu_sensor"].data.pos_w, self.scene["imu_sensor"].data.quat_w
        self.body_ori = get_euler_angle(self.body_ori)
        self.angular_velocities = self.scene["imu_sensor"].data.ang_vel_b
        self.joint_pos = self.scene["robot"].data.joint_pos
        self.joint_vel = self.scene["robot"].data.joint_vel

        # 获取时间和时钟信号a
        clock_signal = 2 * torch.pi * self.phase
        self.sine_clock = torch.sin(clock_signal)
        self.cosine_clock = torch.cos(clock_signal)

        current_map = self.scene["Depth_Camera"].data.output['distance_to_image_plane'].reshape(self.agents_num, -1)
        # 拼接出下一时刻状态空间张量，并归一化
        # 添加噪声后的状态
        current_state = torch.concatenate(
            (self.joint_pos,
             self.joint_vel * 0.05,
             self.body_ori * 0.25,
             self.angular_velocities * 0.25,
             self.sine_clock,
             self.cosine_clock,
             self.action,
             self.vel_cmd,
             0.25 * (current_map.clip(0, 6))
             ), dim=1)

        # #——————————————————————获取当前时刻状态结束————————————————————————————————##

        # #——————————————————————获取额外机器人状态————————————————————————————————##
        # 获得机器人body高度 和 Abad 关节角度
        L_foot_pos, R_foot_pos = self.scene["L_imu_sensor"].data.pos_w, self.scene["R_imu_sensor"].data.pos_w

        self.L_foot_forward, self.L_foot_lateral = yaw_transforming(L_foot_pos[:, 0],
                                                                    L_foot_pos[:, 1],
                                                                    self.body_ori[:, 2])
        self.R_foot_forward, self.R_foot_lateral = yaw_transforming(R_foot_pos[:, 0],
                                                                    R_foot_pos[:, 1],
                                                                    self.body_ori[:, 2])

        L_foot_contact_force = self.scene["L_contact_sensor"].data.net_forces_w[:, 0, 2].view(-1, 1)
        R_foot_contact_force = self.scene["R_contact_sensor"].data.net_forces_w[:, 0, 2].view(-1, 1)
        self.L_foot_contact_situation = L_foot_contact_force > 1e-5
        self.R_foot_contact_situation = R_foot_contact_force > 1e-5
        # #——————————————————————获取额外机器人状态结束————————————————————————————————##
        return current_state

    def get_next_observations(self):
        """
        通用：获取机器人下一时刻状态的函数，包含了所有RL任务都需要的状态信息； 记为St+1。返回什么是Task Dependent
        :return:
        """

        # #——————————————————————获取下一时刻状态————————————————————————————————##
        # 获取机器人的关节身体信息
        self.next_body_pos, self.next_body_ori = self.scene["imu_sensor"].data.pos_w, self.scene[
            "imu_sensor"].data.quat_w
        self.next_body_ori = get_euler_angle(self.next_body_ori)
        self.next_angular_velocities = self.scene["imu_sensor"].data.ang_vel_b
        self.next_joint_pos = self.scene["robot"].data.joint_pos
        self.next_joint_vel = self.scene["robot"].data.joint_vel

        # 获取时间和时钟信号
        self.time += self.dt
        period = 1
        offset = 0.5
        self.phase = self.time % period / period
        clock_signal = 2 * torch.pi * self.phase
        self.sine_clock = torch.sin(clock_signal)
        self.cosine_clock = torch.cos(clock_signal)

        next_map = self.scene["Depth_Camera"].data.output['distance_to_image_plane'].reshape(self.agents_num, -1)
        # 生成噪声

        # 拼接出下一时刻状态空间张量，并归一化
        next_state = torch.concatenate((self.next_joint_pos,
                                        self.next_joint_vel * 0.05,
                                        self.next_body_ori * 0.25,
                                        self.next_angular_velocities * 0.25,
                                        self.sine_clock,
                                        self.cosine_clock,
                                        self.action,
                                        self.vel_cmd,
                                        0.25 * (next_map.clip(0, 6))), dim=1)

        # #——————————————————————获取下一时刻状态结束————————————————————————————————##

        # #——————————————————————获取额外机器人状态————————————————————————————————##
        # 获得机器人body高度 和 Abad 关节角度
        self.next_body_height = self.next_body_pos[:, 2].view(-1, 1)
        L_foot_pos, R_foot_pos = (self.scene["L_imu_sensor"].data.pos_w,
                                  self.scene["R_imu_sensor"].data.pos_w)
        self.next_L_foot_angle, self.next_R_foot_angle = (
            get_euler_angle(self.scene["L_imu_sensor"].data.quat_w)[:, 1:2],
            get_euler_angle(self.scene["R_imu_sensor"].data.quat_w)[:, 1:2])

        self.next_L_foot_lin_vel, self.next_R_foot_lin_vel = (self.scene["L_imu_sensor"].data.lin_vel_b,
                                                              self.scene["R_imu_sensor"].data.lin_vel_b)

        self.next_L_foot_ang_vel, self.next_R_foot_ang_vel = (self.scene["L_imu_sensor"].data.ang_vel_b,
                                                              self.scene["R_imu_sensor"].data.ang_vel_b)

        self.next_L_foot_forward, self.next_L_foot_lateral = yaw_transforming(L_foot_pos[:, 0],
                                                                              L_foot_pos[:, 1],
                                                                              self.next_body_ori[:, 2])
        self.next_R_foot_forward, self.next_R_foot_lateral = yaw_transforming(R_foot_pos[:, 0],
                                                                              R_foot_pos[:, 1],
                                                                              self.next_body_ori[:, 2])
        self.next_L_foot_z = L_foot_pos[:, 2].view(-1, 1)
        self.next_R_foot_z = R_foot_pos[:, 2].view(-1, 1)

        self.next_min_foot_z = torch.minimum(self.next_L_foot_z, self.next_R_foot_z)

        self.next_linear_vel = self.scene["robot"].data.root_lin_vel_w
        L_foot_contact_force = self.scene["L_contact_sensor"].data.net_forces_w[:, 0, 2].view(-1, 1)
        R_foot_contact_force = self.scene["R_contact_sensor"].data.net_forces_w[:, 0, 2].view(-1, 1)

        self.next_L_foot_contact_situation = L_foot_contact_force > 1
        self.next_R_foot_contact_situation = R_foot_contact_force > 1
        self.L_feet_air_time += self.dt * (~self.next_L_foot_contact_situation)
        self.R_feet_air_time += self.dt * (~self.next_R_foot_contact_situation)
        # #——————————————————————获取额外机器人状态结束————————————————————————————————##
        return next_state

    def get_privilege(self):
        """
        Task Dependent：用于特权学习， 获取现实中无法测量的状态
        :return:
        """

        # #——————————————————————获取额外机器人状态————————————————————————————————##
        linear_vel = self.scene["robot"].data.root_lin_vel_w
        L_foot_contact_force = self.scene["L_contact_sensor"].data.net_forces_w[:, 0, 2].view(-1, 1)
        R_foot_contact_force = self.scene["R_contact_sensor"].data.net_forces_w[:, 0, 2].view(-1, 1)
        # #——————————————————————获取额外机器人状态结束————————————————————————————————##
        return torch.concatenate((self.external_body_force/100,
                                          linear_vel/2,
                                          L_foot_contact_force/500,
                                          R_foot_contact_force/500),
                                         dim=1)

    """更新环境"""

    def update_world(self, scaled_action: torch.Tensor):
        """
        通用：更新环境状态
        """
        self.action = scaled_action.clone()
        self.prev_action = scaled_action.clone()
        
        for decimation in range(self.sub_step):
            self.append_action_history(self.action)
            self.real_action = self.action_history[self.all_agent_indices, self.action_delay_idx]
            joint_pos = self.scene["robot"].data.joint_pos
            joint_vel = self.scene["robot"].data.joint_vel
            torque = (self.real_action + self.default_PD_angle - joint_pos) * self.Kp - joint_vel * self.Kd
            torque[:, :6] = torque[:, :6].clip(-80, 80)
            torque[:, -2:] = torque[:, -2:].clip(-20, 20)
            self.scene["robot"].set_joint_effort_target(torque)
            self.scene.write_data_to_sim()
            if self.headless:
                render = False
            else:
                if decimation == self.sub_step - 1:
                    render = True
                else:
                    render = False
            self.sim.step(render=render)
            self.scene.update(self.dt / self.sub_step)

    """-------------------以上均为环境运行代码-----------------------"""
    """-------------------以上均为环境运行代码-----------------------"""
    """-------------------以上均为环境运行代码-----------------------"""

    """-------------------以下均为奖励计算代码-----------------------"""
    """-------------------以下均为奖励计算代码-----------------------"""
    """-------------------以下均为奖励计算代码-----------------------"""

    """-------------------注意！！！计算奖励的代码都不是通用的-----------------------"""
    """-------------------注意！！！计算奖励的代码都不是通用的-----------------------"""
    """-------------------注意！！！单个计算奖励的代码都不是通用的，但是返回reward的函数是通用的-----------------------"""

    def vel_tracking_reward(self):
        """速度跟踪"""

        vel_forward, vel_lateral = yaw_transforming(self.next_linear_vel[:, 0],
                                                    self.next_linear_vel[:, 1],
                                                    self.next_body_ori[:, 2])

        if_forward = (self.vel_cmd == 1).float()

        reward_vel_forward = - 1 * torch.abs(vel_forward - 0.7 * if_forward) + 0.7
        reward_vel_lateral = - 0.6 * torch.abs(vel_lateral)
        reward_vel_vertical = - 0.6 * torch.abs(self.next_linear_vel[:, 2].view(-1, 1))
        reward = reward_vel_lateral + reward_vel_forward + reward_vel_vertical
        reward *= 1
        return reward



    def body_height_tracking_reward(self):
        """高度跟踪"""
        below_min = torch.abs(self.next_body_height - 0.85)

        return -1 * below_min + 0.3



    def body_ori_tracking_reward(self):
        """方位角跟踪（ZYX）"""
        reward_ori_track = -0.0 * self.next_body_ori[:, :2].norm(dim=1, keepdim=True)
        reward_ori_track += -0.3 * self.next_body_ori[:, 2].view(-1, 1).norm(dim=1, keepdim=True)
        reward_ori_track += -0.3 * (self.next_L_foot_angle.abs() + self.next_R_foot_angle.abs()).norm(dim=1,
                                                                                                      keepdim=True)
        reward_ori_track += -0.05 * self.next_angular_velocities.norm(dim=1, keepdim=True)
        reward_ori_track += 0.5
        reward_ori_track *= 1
        return reward_ori_track



    def foot_constraint_reward(self):
        """腿部运动范围限制"""
        foot_regularization_reward = -0.3 * (self.next_joint_pos[:, 0].view(-1, 1) - 0.1).abs()
        foot_regularization_reward += -0.3 * (self.next_joint_pos[:, 1].view(-1, 1) + 0.1).abs()
        foot_regularization_reward += -2 * (self.next_L_foot_z - 0.12).abs() * (~self.next_L_foot_contact_situation)
        foot_regularization_reward += -2 * (self.next_R_foot_z - 0.12).abs() * (~self.next_R_foot_contact_situation)

        foot_regularization_reward += -0.2 * (self.next_L_foot_lin_vel.abs()).sum(dim=-1, keepdim=True) * (
            self.next_L_foot_contact_situation)
        foot_regularization_reward += -0.2 * (self.next_R_foot_lin_vel.abs()).sum(dim=-1, keepdim=True) * (
            self.next_R_foot_contact_situation)

        foot_regularization_reward *= 2

        return foot_regularization_reward


    def single_support_reward(self):
        """
        设定步态
        :return:
        """
        offset = 0.5
        phase_L = self.phase.clone()
        phase_R = (phase_L + offset) % 1

        is_stance_L = phase_L < 0.6
        is_stance_R = phase_R < 0.6
        is_double_stance = is_stance_L & is_stance_R

        """惩罚双脚着地，鼓励单脚着地"""
        single_support = self.next_L_foot_contact_situation != self.next_R_foot_contact_situation
        double_support = self.next_L_foot_contact_situation & self.next_R_foot_contact_situation
        flying = (~self.next_L_foot_contact_situation) & (~self.next_R_foot_contact_situation)

        walking_phase_reward = 0.3 * single_support.float()
        walking_phase_reward += -0.3 * flying.float()
        walking_phase_reward += -0.3 * double_support.float() * (~is_double_stance)

        """鼓励按照我设定的规律步态走"""
        walking_phase_reward += 0.3 * (
            ~(is_stance_L ^ self.next_L_foot_contact_situation)).float() - 0.5  # same, then plus
        walking_phase_reward += 0.3 * (~(is_stance_R ^ self.next_R_foot_contact_situation)).float() - 0.5

        walking_phase_reward *= 1 * (self.vel_cmd == 1)
        return walking_phase_reward.view(-1, 1)

    def foot_air_time_reward(self):
        """鼓励单脚悬空时间，免得快速踏步"""
        L_touching_ground = self.next_L_foot_contact_situation & (~self.L_foot_contact_situation)
        R_touching_ground = self.next_R_foot_contact_situation & (~self.R_foot_contact_situation)

        foot_air_reward = ((self.L_feet_air_time - 0.5) * L_touching_ground +
                           (self.R_feet_air_time - 0.5) * R_touching_ground)

        feet_air_time = 0.2 * foot_air_reward

        self.L_feet_air_time *= (~self.next_L_foot_contact_situation)
        self.R_feet_air_time *= (~self.next_R_foot_contact_situation)
        feet_air_time *= 5 * (self.vel_cmd == 1)
        return feet_air_time.view(-1, 1)

    def stand_still(self):
        """当你收到静止指令时，惩罚身体和腿部的运动，鼓励机器人站立不动"""
        stand_still_reward = -0.5 * self.joint_pos[:, 2:].abs().mean(dim=-1, keepdim=True)
        stand_still_reward += -0.1 * self.joint_vel.abs().mean(dim=-1, keepdim=True)
        stand_still_reward *= (self.vel_cmd == 0)
        return stand_still_reward

    def Termination_reward(self):
        """设定结束的条件"""
        over1 = torch.abs(self.next_body_ori[:, 0].view(-1, 1)) > np.pi / 3
        over2 = torch.abs(self.next_body_ori[:, 1].view(-1, 1)) > np.pi / 3
        over3 = torch.abs(self.next_body_ori[:, 2].view(-1, 1)) > np.pi / 3
        over4 = (self.next_body_height - self.next_min_foot_z) < 0.7
        over5 = self.next_min_foot_z < 0.05
        self.over = over1 | over2 | over3 | over4 | over5
        reward_fall = - self.over.float() * 10
        return reward_fall.view(-1, 1)

    def compute_reward(self):
        """
        通用：计算你上面设置的所有奖励的合成奖励，并且进行记录用于你print
        :return:
        """
        reward = 0
        reward += 1 * self.vel_tracking_reward()
        reward += 1 * self.body_height_tracking_reward()
        reward += 1 * self.body_ori_tracking_reward()
        reward += 1 * self.foot_constraint_reward()
        reward += 1 * self.single_support_reward()
        reward += 1 * self.foot_air_time_reward()
        reward += 1 * self.stand_still()
        reward += 1 * self.Termination_reward()
        reward += 0.8

        self.vel_tracking_reward_sum += self.vel_tracking_reward().mean().item() / self.max_step
        self.body_height_tracking_reward_sum += self.body_height_tracking_reward().mean().item() / self.max_step
        self.body_ori_tracking_reward_sum += self.body_ori_tracking_reward().mean().item() / self.max_step
        self.foot_constraint_reward_sum += self.foot_constraint_reward().mean().item() / self.max_step
        self.stand_still_reward_sum += self.stand_still().mean().item() / self.max_step
        self.single_support_reward_sum += self.single_support_reward().mean().item() / self.max_step
        self.foot_air_time_reward_sum += self.foot_air_time_reward().mean().item() / self.max_step
        self.Termination_reward_sum += self.Termination_reward().mean().item() / self.max_step

        return reward, self.over.float(), (self.time > 20).float()

    def print_reward_sum(self):
        """
        想加就加
        :return:
        """
        print(f"vel_tracking_reward_sum: {self.vel_tracking_reward_sum:.4f}")
        print(f"body_height_tracking_reward_sum: {self.body_height_tracking_reward_sum:.4f}")
        print(f"body_ori_tracking_reward_sum: {self.body_ori_tracking_reward_sum:.4f}")
        print("")
        print(f"foot_constraint_reward_sum: {self.foot_constraint_reward_sum:.4f}")
        print(f"stand_still_reward_sum: {self.stand_still_reward_sum:.4f}")
        print("")
        print(f"single_support_reward_sum: {self.single_support_reward_sum:.4f}")
        print(f"foot_air_time_reward_sum: {self.foot_air_time_reward_sum:.4f}")
        print("")
        print(f"Termination_reward_sum: {self.Termination_reward_sum:.4f}")
        print(f"simulation time:{self.t_module.time() - self.start_time:.2f} s")

        self.vel_tracking_reward_sum = 0
        self.body_height_tracking_reward_sum = 0
        self.body_ori_tracking_reward_sum = 0
        self.foot_constraint_reward_sum = 0
        self.stand_still_reward_sum = 0
        self.single_support_reward_sum = 0
        self.foot_air_time_reward_sum = 0
        self.Termination_reward_sum = 0
