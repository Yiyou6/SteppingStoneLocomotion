from SF_TRON_FP.SRC.PPO.Buffer import *

""" PPO算法的Actor-Critic网络结构
Actor: 输入状态 输出动作的均值和标准差"""


class BaseNetwork(torch.nn.Module):
    def __init__(self, state_dim: int, num_neurons: int):
        """
        创建一套基础的神经网络，给Actor和Critic共享
        :param state_dim: 你的状态维度 ，例如一个倒立摆的状态有theta和theta_dot两个维度，那么state_dim就是2
        :param num_neurons: 每层用到的神经元数量，随你怎么填，1024以下即可，大了边际效应低
        """
        super(BaseNetwork, self).__init__()
        self.state_dim = state_dim
        self.num_neurons = num_neurons

        # 共享的主干网络
        self.fc1_x = torch.nn.Linear(self.state_dim, self.num_neurons * 4)
        self.fc2_x = torch.nn.Linear(self.num_neurons * 4, self.num_neurons * 2)
        self.fc3_x = torch.nn.Linear(self.num_neurons * 2, self.num_neurons)

    def process_input(self, input_: torch.Tensor):
        """
        :param input_: 必须是torch tensor，形状是[agent_num, state_dim]，例如你有16个机器人，每个机器人的状态维度是10，那么输入就是[16, 10]
        :return: 处理后的特征，形状是[agent_num, num_neurons]，例如你有16个机器人，每个机器人的特征维度是256，那么输出就是[16, 256]
        """
        x = input_
        # 通过主干网络
        x = torch.nn.functional.elu(self.fc1_x(x))
        x = torch.nn.functional.elu(self.fc2_x(x))
        x = torch.nn.functional.elu(self.fc3_x(x))

        return x


class Actor(BaseNetwork):
    def __init__(self, state_dim: int, num_neurons: int, actuator_num: int, action_scale: int = 1, std_scale: int = 0.5):
        """
        Actor类提供了采样action的作用，这里是连续动作空间的Actor网络，输出动作的均值和标准差（采样用正态分布）
        :param state_dim: 状态维度
        :param num_neurons: 每层用到的神经元数量，随你怎么填，1024以下即可，大了边际效应低
        :param actuator_num: 你有几个执行器？ 例如机器人有8个电机，那就是8
        :param action_scale: Actor的均值的默认输出范围是【-1，1】，action_scale = 2则使输出范围会放大两倍，即【-2，2】
        :param std_scale: 标准差的初始大小，默认为0.5. 不要太大，大了训练效果垃圾因为太随机了。 注意标准差的值同样是会被训练的
        """
        super(Actor, self).__init__(state_dim, num_neurons)
        self.actuator_num = actuator_num
        self.act_scale = action_scale
        self.std_scale = std_scale

        self.mean = torch.nn.Linear(self.num_neurons, self.actuator_num)
        self.std = torch.nn.Parameter(torch.ones(self.actuator_num))

    def forward(self, input_: torch.Tensor):
        """
        :param input_: 必须是torch tensor，形状是[agent_num, state_dim]，例如你有16个机器人，每个机器人的状态维度是10，那么输入就是[16, 10]
        注意，在训练过程中，输入的状态可能是[step, agent_num,state_dim]，但这不影响什么东西。详情可以自己了解
        :return: 返回动作的均值和标准差，都是torch tensor，形状是[agent_num, actuator_num]，例如你有16个机器人，每个机器人有8个电机，那么输出就是[16, 8]
        """
        x = self.process_input(input_)
        mu = torch.nn.functional.tanh(self.mean(x))
        std = torch.abs(self.std_scale * self.std)
        return mu, std


class Critic(BaseNetwork):
    """
    Critic类就是负责输入状态和输出对应的state value
    """
    def __init__(self, state_dim, num_neurons):
        super(Critic, self).__init__(state_dim, num_neurons)
        self.fc4_x = torch.nn.Linear(self.num_neurons, 1)

    def forward(self, input_):
        """

        :param input_: 须是torch tensor，形状是[agent_num, state_dim]，例如你有16个机器人，每个机器人的状态维度是10，那么输入就是[16, 10]
        注意：同上
        :return: 回状态的价值，都是torch tensor，形状是[agent_num, 1]，例如你有16个机器人，那么输出就是[16, 1]
        """

        x = self.process_input(input_)
        x = self.fc4_x(x)
        return x


""" Actor-Critic类 包含Actor和Critic网络 以及相关的优化器和经验回放缓冲区"""


class Actor_Critic:
    def __init__(self, PPOCfg, EnvCfg, index=0):
        """ 初始化Actor-Critic网络
        Args:
            PPOCfg: PPO算法的配置参数 (类型: 配置类)
            EnvCfg: 环境的配置参数 (类型: 配置类)
            index: 该AC的索引，保存神经网络时会用这个索引作为名字，且读取的时候也会读取对应的索引 (类型: int, 默认值: 0)
        """
        self.index = index
        # PPO parameter
        self.gamma = PPOCfg.PPOParam.gamma
        self.lam = PPOCfg.PPOParam.lam
        self.epsilon = PPOCfg.PPOParam.epsilon
        self.policy_smooth = PPOCfg.PPOParam.policy_smooth # 这个不是强化学习的一部分，只是一个保证输出动作平滑的损失项的系数，越大越平滑，但过大可能会影响性能，自己调试
        self.entropy_coef = PPOCfg.PPOParam.entropy_coef
        self.batch_size = PPOCfg.PPOParam.batch_size
        self.loss_fn = torch.nn.MSELoss()

        # state parameter
        self.state_dim = PPOCfg.CriticParam.state_dim
        self.critic_num_layers = PPOCfg.CriticParam.critic_layers_num
        self.critic_update_frequency = PPOCfg.CriticParam.critic_update_frequency
        self.critic_lr = PPOCfg.CriticParam.critic_lr

        # Env parameter
        self.agent_num = EnvCfg.EnvParam.agents_num
        self.device = EnvCfg.EnvParam.device
        self.maximum_step = PPOCfg.PPOParam.maximum_step
        self.train = EnvCfg.EnvParam.train

        # actor parameter
        self.action_scale = torch.FloatTensor(PPOCfg.ActorParam.action_scale).to(self.device)
        self.std_scale = PPOCfg.ActorParam.std_scale
        self.actor_num_layers = PPOCfg.ActorParam.act_layers_num
        self.actor_update_frequency = PPOCfg.ActorParam.actor_update_frequency
        self.actuator_num = PPOCfg.ActorParam.actuator_num
        self.actor_lr = PPOCfg.ActorParam.actor_lr

        if self.batch_size > (self.agent_num * self.maximum_step):
            self.batch_size = self.agent_num * self.maximum_step

        # 初始化网络
        self.actor = Actor(self.state_dim,
                           self.actor_num_layers,
                           self.actuator_num,
                           self.action_scale,
                           self.std_scale).to(self.device)

        self.critic = Critic(self.state_dim,
                             self.critic_num_layers).to(self.device)

        # 初始化优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.Buffer = Agent_State_Buffer(self.state_dim,
                                         self.actuator_num,
                                         self.agent_num,
                                         self.maximum_step,
                                         self.device)

        self.idx = [torch.randperm(self.maximum_step * self.agent_num, device=self.device)[:self.batch_size]
                    for _ in range(self.critic_update_frequency)]

        self.initial_reward_sum = -999

    def sample_action(self, state, deterministic=True):
        """
        根据状态获取动作 用于与环境交互
        Args:
            state: 当前状态 (类型: torch.tensor, 形状: [agent_num, state_dim])
            deterministic: 是否使用确定性策略
        Returns:
            action: 选择的action输出值 (类型: torch.tensor, 形状: [agent_num, actuator_num])
        """
        with torch.no_grad():
            mu, std = self.actor(state)

        if deterministic:
            action = mu

        else:
            action = torch.normal(mu, std).clip(-1, 1)
        return action, action * self.action_scale

    def store_experience(self, state, action, next_state, reward, over, current_step):
        """ 存储经验到缓冲区
        Args:
            state: 当前状态 (类型: torch.tensor, 形状: [agent_num, state_dim])
            action: 当前动作 (类型: torch.tensor, 形状: [agent  _num, actuator_num])
            next_state: 当前状态 (类型: torch.tensor, 形状: [agent_num, state_dim])
            reward: 当前奖励 (类型: torch.tensor, 形状: [agent_num, 1])
            over: 当前是否结束 (类型: torch.tensor, 形状: [agent_num, 1])
            current_step: 当前时间步 (类型: int)
        """
        self.Buffer.store_state(state, current_step)
        self.Buffer.store_action(action, current_step)
        self.Buffer.store_next_state(next_state, current_step)
        self.Buffer.store_reward(reward, current_step)
        self.Buffer.store_over(over, current_step)

    def update(self):

        # 获取经验数据
        buffer = self.Buffer
        state = buffer.state_buffer.view(-1, self.state_dim)
        action = buffer.action_buffer.view(-1, self.actuator_num)
        next_state = buffer.next_state_buffer.view(-1, self.state_dim)
        reward = buffer.reward_buffer.view(-1, 1)
        over = buffer.over_buffer.view(-1, 1)
        reward_sum = reward.mean().item()

        self.save_each_epi_model()
        if reward_sum > self.initial_reward_sum:
            self.initial_reward_sum = reward_sum
            self.save_best_model()
            print(f"Best Model Saved")

        # 计算旧策略概率
        with torch.no_grad():
            mu_old, std_old = self.actor(state)
            old_prob = torch.distributions.Normal(mu_old, std_old).log_prob(action).sum(dim=1, keepdim=True)

        # Critic更新
        for i in range(self.critic_update_frequency):
            idx = self.idx[i]
            s_batch, ns_batch, r_batch, o_batch = state[idx], next_state[idx], reward[idx], over[idx]

            value = self.critic(s_batch)
            with torch.no_grad():
                next_value = self.critic(ns_batch)
                target_value = r_batch + self.gamma * next_value * (1 - o_batch)
            critic_loss = self.loss_fn(value, target_value) + 0 * self.loss_fn(value, next_value)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # 计算GAE
        buffer.compute_GAE(self.critic, self.gamma, self.lam)
        GAE = buffer.GAE_buffer.view(-1, 1)

        # Actor更新
        for i in range(self.actor_update_frequency):
            idx = self.idx[i]
            s_batch, a_batch, ns_batch = state[idx], action[idx], next_state[idx]
            gae_batch, old_prob_batch = GAE[idx], old_prob[idx]

            # 计算新策略
            mu, std = self.actor(s_batch)
            with torch.no_grad():
                next_mu, _ = self.actor(ns_batch)
            new_prob = torch.distributions.Normal(mu, std).log_prob(a_batch).sum(dim=1, keepdim=True)

            # PPO损失
            ratio = torch.exp(new_prob - old_prob_batch)  # 括号里是对数
            surr1 = ratio * gae_batch
            surr2 = ratio.clamp(1 - self.epsilon, 1 + self.epsilon) * gae_batch

            entropy = torch.distributions.Normal(mu, std).entropy().mean()
            actor_loss = (-torch.min(surr1, surr2).mean()
                          - self.entropy_coef * entropy
                          + self.policy_smooth * self.loss_fn(mu, next_mu))

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        print("std_mean", std_old.mean())
        print(f"Experience Collected: {len(state)}, Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}")
        print("reward:", reward_sum)

    def save_best_model(self):
        torch.save(self.actor.state_dict(), f'Model/NN_Model/actor{self.index}.pth')
        torch.save(self.critic.state_dict(), f'Model/NN_Model/critic{self.index}.pth')

    def save_each_epi_model(self):
        torch.save(self.actor.state_dict(), f'Model/NN_Model/actor{self.index}_f.pth')
        torch.save(self.critic.state_dict(), f'Model/NN_Model/critic{self.index}_f.pth')

    def load_best_model(self):
        self.actor.load_state_dict(torch.load(f'Model/NN_Model/actor{self.index}.pth'))
        self.critic.load_state_dict(torch.load(f'Model/NN_Model/critic{self.index}.pth'))

    def load_each_epi_model(self):
        self.actor.load_state_dict(torch.load(f'Model/NN_Model/actor{self.index}_f.pth'))
        self.critic.load_state_dict(torch.load(f'Model/NN_Model/critic{self.index}_f.pth'))
