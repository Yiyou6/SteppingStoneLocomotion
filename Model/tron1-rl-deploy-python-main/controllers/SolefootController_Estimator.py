import copy
import numpy as np
import yaml
import time
import onnxruntime as ort
from functools import partial
import limxsdk
import limxsdk.robot.Rate as Rate
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import limxsdk.datatypes as datatypes
import rospy
from sensor_msgs.msg import Image  # ROS图像消息类型


class SolefootController_Estimator:
    def __init__(self, model_dir, robot, robot_type, rl_type, start_controller):

        import signal
        import sys

        # 初始化running标志
        self.running = True

        # 设置信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)

        # Initialize robot and type information
        self.robot = robot
        self.robot_type = robot_type
        self.rl_type = rl_type

        # Load configuration and Model file paths based on robot type
        self.config_file = f'{model_dir}/{self.robot_type}/params.yaml'
        self.model1 = f'{model_dir}/{self.robot_type}/policy/{self.rl_type}/model1.onnx'
        self.model2 = f'{model_dir}/{self.robot_type}/policy/{self.rl_type}/model2.onnx'
        self.model3 = f'{model_dir}/{self.robot_type}/policy/{self.rl_type}/model3.onnx'
        # Load configuration settings from the YAML file

        self.default_PD_angle = np.array([0.15, -0.15,
                                          0., -0.,
                                          -0, 0.,
                                          -0., -0.])

        self.load_config(self.config_file)
        # Load the ONNX Model
        self.initialize_onnx_models()

        # Prepare robot command structure with default values for mode, q, dq, tau, Kp, Kd
        self.robot_cmd = datatypes.RobotCmd()
        self.robot_cmd.mode = [0. for _ in range(0, self.joint_num)]
        self.robot_cmd.q = [0. for _ in range(0, self.joint_num)]
        self.robot_cmd.dq = [0. for _ in range(0, self.joint_num)]
        self.robot_cmd.tau = [0. for _ in range(0, self.joint_num)]
        self.robot_cmd.Kp = [self.control_cfg['stiffness'] for _ in range(0, self.joint_num)]
        self.robot_cmd.Kd = [self.control_cfg['damping'] for _ in range(0, self.joint_num)]

        # Prepare robot state structure
        self.robot_state = datatypes.RobotState()
        self.robot_state.tau = [0. for _ in range(0, self.joint_num)]
        self.robot_state.q = [0. for _ in range(0, self.joint_num)]
        self.robot_state.dq = [0. for _ in range(0, self.joint_num)]
        self.robot_state_tmp = copy.deepcopy(self.robot_state)

        # Initialize IMU (Inertial Measurement Unit) data structure
        self.imu_data = datatypes.ImuData()
        self.imu_data.quat[0] = 1
        self.imu_data.quat[1] = 0
        self.imu_data.quat[2] = 0
        self.imu_data.quat[3] = 0
        self.yaw_offset = 0
        self.sim_time = 0
        self.imu_data_tmp = copy.deepcopy(self.imu_data)

        # Set up a callback to receive updated robot state data
        self.robot_state_callback_partial = partial(self.robot_state_callback)
        self.robot.subscribeRobotState(self.robot_state_callback_partial)

        # Set up a callback to receive updated IMU data
        self.imu_data_callback_partial = partial(self.imu_data_callback)
        self.robot.subscribeImuData(self.imu_data_callback_partial)

        # Set up a callback to receive updated SensorJoy
        self.sensor_joy_callback_partial = partial(self.sensor_joy_callback)
        self.robot.subscribeSensorJoy(self.sensor_joy_callback_partial)

        # Set up a callback to receive diagnostic data
        self.robot_diagnostic_callback_partial = partial(self.robot_diagnostic_callback)
        self.robot.subscribeDiagnosticValue(self.robot_diagnostic_callback_partial)

        #set up stupid camera
        rospy.init_node('solefoot_controller', anonymous=True)
        self.setup_depth_camera_subscriber()

        # Initialize the calibration state to -1, indicating no calibration has occurred.
        self.calibration_state = -1

        # Flag to start the controller
        self.start_controller = start_controller

    def initialize_onnx_models(self):
        # Configure ONNX Runtime session options to optimize CPU usage
        session_options = ort.SessionOptions()
        # Limit the number of threads used for parallel computation within individual operators
        session_options.intra_op_num_threads = 1
        # Limit the number of threads used for parallel execution of different operators
        session_options.inter_op_num_threads = 1
        # Enable all possible graph optimizations to improve inference performance
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Disable CPU memory arena to reduce memory fragmentation
        session_options.enable_cpu_mem_arena = False
        # Disable memory pattern optimization to have more control over memory allocation
        session_options.enable_mem_pattern = False

        # Define execution providers to use CPU only, ensuring no GPU inference
        cpu_providers = ['CPUExecutionProvider']

        # Load the ONNX Model and set up input and output names
        self.policy_session1 = ort.InferenceSession(self.model1, sess_options=session_options,
                                                    providers=cpu_providers)
        self.policy_input_names1 = [self.policy_session1.get_inputs()[i].name for i in
                                    range(self.policy_session1.get_inputs().__len__())]
        self.policy_output_names1 = [self.policy_session1.get_outputs()[i].name for i in
                                     range(self.policy_session1.get_outputs().__len__())]
        self.policy_input_shapes1 = [self.policy_session1.get_inputs()[i].shape for i in
                                     range(self.policy_session1.get_inputs().__len__())]
        self.policy_output_shapes1 = [self.policy_session1.get_outputs()[i].shape for i in
                                      range(self.policy_session1.get_outputs().__len__())]

        self.policy_session2 = ort.InferenceSession(self.model2, sess_options=session_options,
                                                    providers=cpu_providers)
        self.policy_input_names2 = [self.policy_session2.get_inputs()[i].name for i in
                                    range(self.policy_session2.get_inputs().__len__())]
        self.policy_output_names2 = [self.policy_session2.get_outputs()[i].name for i in
                                     range(self.policy_session2.get_outputs().__len__())]
        self.policy_input_shapes2 = [self.policy_session2.get_inputs()[i].shape for i in
                                     range(self.policy_session2.get_inputs().__len__())]
        self.policy_output_shapes2 = [self.policy_session2.get_outputs()[i].shape for i in
                                      range(self.policy_session2.get_outputs().__len__())]

        self.policy_session3 = ort.InferenceSession(self.model3, sess_options=session_options,
                                                    providers=cpu_providers)
        self.policy_input_names3 = [self.policy_session3.get_inputs()[i].name for i in
                                    range(self.policy_session3.get_inputs().__len__())]
        self.policy_output_names3 = [self.policy_session3.get_outputs()[i].name for i in
                                     range(self.policy_session3.get_outputs().__len__())]
        self.policy_input_shapes3 = [self.policy_session3.get_inputs()[i].shape for i in
                                     range(self.policy_session3.get_inputs().__len__())]
        self.policy_output_shapes3 = [self.policy_session3.get_outputs()[i].shape for i in
                                      range(self.policy_session3.get_outputs().__len__())]

    # Load the configuration from a YAML file
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Assign configuration parameters to controller variables
        self.joint_names = config['PointfootCfg']['joint_names']
        self.init_state = config['PointfootCfg']['init_state']['default_joint_angle']
        self.stand_duration = config['PointfootCfg']['stand_mode']['stand_duration']
        self.control_cfg = config['PointfootCfg']['control']
        self.actions_size = config['PointfootCfg']['size']['actions_size']
        self.action_scale = np.array([1, 1, 1, 1, 1, 1, 1, 1])

        self.observations_size = config['PointfootCfg']['size']['observations_size']

        self.imu_orientation_offset = np.array(list(config['PointfootCfg']['imu_orientation_offset'].values()))

        self.loop_frequency = config['PointfootCfg']['loop_frequency']

        # Initialize variables for actions, observations, and commands
        self.actions = np.zeros(self.actions_size)
        self.actions1 = np.zeros(self.actions_size)
        self.actions2 = np.zeros(self.actions_size)
        self.observations = np.zeros(self.observations_size)

        self.loop_count = 0  # loop iteration count
        self.stand_percent = 0  # percentage of time the robot has spent in stand mode
        self.policy_session = None  # ONNX Model session for policy inference
        self.joint_num = len(self.joint_names)  # number of joints

        self.ankle_joint_stiffness = config['PointfootCfg']['control']['ankle_joint_stiffness']
        self.ankle_joint_damping = config['PointfootCfg']['control']['ankle_joint_damping']

        # Initialize joint angles based on the initial configuration
        self.init_joint_angles = np.zeros(len(self.joint_names))
        for i in range(len(self.joint_names)):
            self.init_joint_angles[i] = self.init_state[self.joint_names[i]]

        self.last_actions = self.init_joint_angles
        # Set initial mode to "STAND"
        self.mode = "STAND"

    # Main control loop
    def run(self):
        # Wait until the controller is started
        while not self.start_controller:
            time.sleep(1)

        # Initialize default joint angles for standing
        self.stand_percent += 1 / (self.stand_duration * self.loop_frequency)
        self.mode = "STAND"
        self.loop_count = 0

        # Set the loop rate based on the frequency in the configuration
        rate = Rate(self.loop_frequency)
        self.robot_state_tmp = copy.deepcopy(self.robot_state)
        self.default_joint_angles = np.array(self.robot_state_tmp.q)  # np.array([0.0] * len(self.joint_names))

        while self.start_controller and self.running:
            self.update()
            rate.sleep()

        # Reset robot command values to ensure a safe stop when exiting the loop
        self.robot_cmd.q = [0. for _ in range(0, self.joint_num)]
        self.robot_cmd.dq = [0. for _ in range(0, self.joint_num)]
        self.robot_cmd.tau = [0. for _ in range(0, self.joint_num)]
        self.robot_cmd.Kp = [0. for _ in range(0, self.joint_num)]
        self.robot_cmd.Kd = [3 for _ in range(0, self.joint_num)]
        self.robot.publishRobotCmd(self.robot_cmd)
        time.sleep(1)

    def update(self):
        """
        Updates the robot's state based on the current mode and publishes the robot command.
        """

        if self.mode == "STAND":
            self.handle_stand_mode()
        elif self.mode == "WALK":
            self.handle_walk_mode()

        # Increment the loop count
        self.loop_count += 1

        # Publish the robot command
        self.robot.publishRobotCmd(self.robot_cmd)

    # Handle the stand mode for smoothly transitioning the robot into standing
    def handle_stand_mode(self):
        if self.stand_percent < 1:
            for j in range(len(self.joint_names)):
                # Interpolate between initial and default joint angles during stand mode
                if (j + 1) % 4 != 0:
                    pos_des = self.default_joint_angles[j] * (1 - self.stand_percent) + self.init_state[
                        self.joint_names[j]] * self.stand_percent
                    self.set_joint_command(j, pos_des, 0, 0, 120, 6)  # 恢复站立的时候不要太猛
                else:
                    pos_des = self.default_joint_angles[j] * (1 - self.stand_percent) + self.init_state[
                        self.joint_names[j]] * self.stand_percent
                    self.set_joint_command(j, pos_des, 0, 0, 80, 4)  # 恢复站立的时候不要太猛

            # Increment the stand percentage over time
            self.stand_percent += 1 / (self.stand_duration * self.loop_frequency)
            if self.stand_percent > 1:
                self.stand_percent = 1
                while not self.start:
                    time.sleep(0.1)
        else:

            self.imu_data_tmp = copy.deepcopy(self.imu_data)
            imu_orientation = np.array(self.imu_data_tmp.quat)
            eul_angle = self.get_euler_angle(imu_orientation)
            self.robot_state_tmp = copy.deepcopy(self.robot_state)
            self.yaw_offset = eul_angle[2]
            # Switch to walk mode after standing
            self.mode = "WALK"

    # Handle the walk mode where the robot moves based on computed actions
    def handle_walk_mode(self):
        # Update the temporary robot state and IMU data
        self.robot_state_tmp = copy.deepcopy(self.robot_state)
        self.imu_data_tmp = copy.deepcopy(self.imu_data)
        # Execute actions every 'decimation' iterations
        if self.loop_count % self.control_cfg['decimation'] == 0:
            self.compute_observation()
            self.compute_actions()
            self.sim_time += 0.02

        # Iterate over the joints and set commands based on actions
        joint_pos = np.array(self.robot_state_tmp.q)
        joint_vel = np.array(self.robot_state_tmp.dq)
        for i in range(len(joint_pos)):
            pos_des = self.actions[i]
            if (i + 1) % 4 != 0:

                self.set_joint_command(i, pos_des, 0, 0, self.control_cfg['stiffness'], self.control_cfg['damping'])
            else:
                self.set_joint_command(i, pos_des, 0, 0, self.ankle_joint_stiffness, self.ankle_joint_damping)

    def swap_positions(self, initial_array, reverse=False):
        # Gym顺序: [左腿4关节, 右腿4关节]
        # Lab顺序: [左右abad, 左右hip, 左右knee, 左右ankle]

        # 索引映射
        gym_to_lab = [0, 4, 1, 5, 2, 6, 3, 7]  # Gym索引 -> Lab索引

        new_array = np.zeros_like(initial_array)

        if not reverse:
            # Gym -> Lab
            for i in range(8):
                new_array[i] = initial_array[gym_to_lab[i]]
        else:
            # Lab -> Gym: 需要逆映射
            for i in range(8):
                new_array[gym_to_lab[i]] = initial_array[i]

        return new_array

    def compute_observation(self):
        # Convert IMU orientation from quaternion to Euler angles (ZYX convention)
        imu_orientation = np.array(self.imu_data_tmp.quat)
        euler_angles = self.get_euler_angle(imu_orientation)
        print(euler_angles)

        # Retrieve base angular velocity from the IMU data
        base_ang_vel = np.array(self.imu_data_tmp.gyro)

        # Retrieve joint positions and velocities from the robot state
        joint_positions = np.array(self.robot_state_tmp.q)
        joint_velocities = np.array(self.robot_state_tmp.dq)
        # Retrieve the last actions that were applied to the robot
        last_actions = np.array(self.last_actions).copy()

        # Populate observation vector
        joint_pos_input = (joint_positions - self.init_joint_angles)
        # swap positions in joint_pos, joint_vel and actions if mode is isaaclab
        # gym order 2 lab order， reverse = False means gym 2 lab
        joint_pos_input = self.swap_positions(joint_pos_input, reverse=False)
        joint_velocities = self.swap_positions(joint_velocities, reverse=False)
        last_actions = self.swap_positions(last_actions, reverse=False)
        period = 1
        phase = self.sim_time % period / period
        sine_clock = np.sin(2 * np.pi * phase).reshape(1)
        cosine_clock = np.cos(2 * np.pi * phase).reshape(1)

        if len(self.depth_image) > 400:
            self.depth_image = np.clip(self.downsample_depth_image(self.depth_image).flatten(), 0, 6)

        self.depth_image = self.depth_image.flatten()
        obs = np.concatenate([
            joint_pos_input,  # Scaled joint positions
            joint_velocities * 0.05,  # Scaled joint velocities
            euler_angles * 0.25,  # Scaled base orientation (Euler angles)
            base_ang_vel * 0.25,  # Scaled base angular velocity
            sine_clock,
            cosine_clock,
            last_actions,  # Lab order
            self.cmd,
            self.depth_image * 0.25
        ])

        self.observations = obs.copy()

    def get_euler_angle(self, quat):  # I have checked it, it is correct
        """Convert quaternion to Euler angles (roll, pitch, yaw) in radians.
        Args:
            quat (np.Tensor): Tensor of shape (N, 4) representing quaternions
                                 in the order (x,y,z,w).
        Returns:
            np.Tensor: Tensor of shape (N, 3) representing Euler angles
                          in radians in the order (roll, pitch, yaw).
        """
        quat = quat.reshape(4)
        w = quat[0]
        x = quat[1]
        y = quat[2]
        z = quat[3]

        # Roll (x), Pitch (y), Yaw (z)
        # Using the ZYX convention XYZ Euler

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        # Use 1.0 - 1e-6 to avoid NaN when |sinp| is slightly > 1.0 due to floating point
        sinp = np.clip(sinp, -1.0 + 1e-6, 1.0 - 1e-6)
        pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        angles = np.array([roll, pitch, yaw])

        # Angle adjustments to keep them within [-pi/2, pi/2]
        angles = np.where(angles < -np.pi / 2, angles + np.pi, angles)
        angles = np.where(angles > np.pi / 2, angles - np.pi, angles)
        angles[2] -= self.yaw_offset

        if np.any(np.abs(np.rad2deg(angles)) > 45) and self.stand_percent >= 1 and self.mode == "WALK":
            self.start_controller = False
            print(angles)
            print("pose out of range, stop controller")
        return angles

    def compute_actions(self):
        """
        Computes the actions based on the current observations using the policy session.
        """
        # Concatenate observations into a single tensor and convert to float32
        input_tensor1 = self.observations.reshape(1, -1).copy()
        input_tensor1[0, 33:] = 0
        input_tensor1 = input_tensor1.astype(np.float32)

        # Create a dictionary of inputs for the policy session
        inputs1 = {self.policy_input_names1[0]: input_tensor1}

        # Run the policy session and get the output
        output1, _ = self.policy_session1.run(self.policy_output_names1, inputs1)

        input_tensor2 = self.observations.reshape(1, -1).copy()
        input_tensor2 = input_tensor2.astype(np.float32)

        # Create a dictionary of inputs for the policy session
        inputs2 = {self.policy_input_names2[0]: input_tensor2}

        # Run the policy session and get the output
        output2, _ = self.policy_session2.run(self.policy_output_names2, inputs2)

        # Flatten the output and store it as actions
        self.actions1 = np.array(output1).flatten()  # Lab order
        self.actions2 = np.array(output2).flatten()  # Lab order

        self.actions = self.actions1.copy()
        self.actions += self.default_PD_angle

        self.actions = self.swap_positions(self.actions, reverse=True)  # Lab order to gym order

        self.last_actions = self.actions.copy()

    def set_joint_command(self, joint_index, q, dq, tau, kp, kd):
        """
        Sends a command to configure the state of a specific joint.
        This method updates the joint's desired position, velocity, torque, and control gains.
        Replace this implementation with the actual communication logic for your hardware.

        Parameters:
        joint_index (int): The index of the joint to be controlled.
        q (float): The desired joint position, typically in radians or degrees.
        dq (float): The desired joint velocity, typically in radians/second or degrees/second.
        tau (float): The desired joint torque, typically in Newton-meters (Nm).
        kp (float): The proportional gain for position control.
        kd (float): The derivative gain for velocity control.
        """
        self.robot_cmd.q[joint_index] = q
        self.robot_cmd.dq[joint_index] = dq
        self.robot_cmd.tau[joint_index] = tau
        self.robot_cmd.Kp[joint_index] = kp
        self.robot_cmd.Kd[joint_index] = kd

    # Callback function for receiving robot command data
    def robot_state_callback(self, robot_state: datatypes.RobotState):
        """
        Callback function to update the robot state from incoming data.

        Parameters:
        robot_state (datatypes.RobotState): The current state of the robot.
        """
        self.robot_state = robot_state

    # Callback function for receiving imu data
    def imu_data_callback(self, imu_data: datatypes.ImuData):
        """
        Callback function to update IMU data from incoming data.

        Parameters:
        imu_data (datatypes.ImuData): The IMU data containing stamp, acceleration, gyro, and quaternion.
        """
        self.imu_data.stamp = imu_data.stamp
        self.imu_data.acc = imu_data.acc
        self.imu_data.gyro = imu_data.gyro
        self.imu_data.quat = imu_data.quat

    # Callback function for receiving sensor joy data
    def sensor_joy_callback(self, sensor_joy: datatypes.SensorJoy):
        # Check if the robot is in the calibration state and both L1 (button index 4) and Y (button index 3) buttons are pressed.
        if not self.start_controller and self.calibration_state == 0 and sensor_joy.buttons[4] == 1 and \
                sensor_joy.buttons[3] == 1:
            print(f"L1 + Y: start_controller...")
            self.start_controller = True

        # Check if both L1 (button index 4) and X (button index 2) are pressed to stop the controller
        if self.start_controller and sensor_joy.buttons[4] == 1 and sensor_joy.buttons[2] == 1:
            print(f"L1 + X: stop_controller...")
            self.start_controller = False

        linear_x = sensor_joy.axes[1]
        linear_x = 1.0 if linear_x > 0.5 else 0.0

        linear_y = sensor_joy.axes[2]
        self.cmd = np.array([linear_x])
        linear_x = sensor_joy.axes[1]
        self.start = True if linear_x < -0.5 else False
        self.start_controller = False if abs(linear_y) > 0.5 else True
        print(linear_y)

    # Callback function for receiving diagnostic data
    def robot_diagnostic_callback(self, diagnostic_value: datatypes.DiagnosticValue):
        # Check if the received diagnostic data is related to calibration.
        if diagnostic_value.name == "calibration":
            print(f"Calibration state: {diagnostic_value.code}")
            self.calibration_state = diagnostic_value.code

    def setup_depth_camera_subscriber(self):
        """
        设置深度相机数据订阅器
        """
        # 深度图像话题名称（需要根据实际ROS配置修改）
        depth_topic = "/camera0/depth/image_rect_raw"

        # 订阅深度图像话题
        rospy.Subscriber(depth_topic, Image,
                         self.depth_image_callback, queue_size=1)

        rospy.loginfo(f"深度相机订阅已启动，话题: {depth_topic}")

    def depth_image_callback(self, msg):
        """
        深度图像回调函数

        参数:
        msg: sensor_msgs/Image 消息，包含深度图像数据
        """
        try:
            # 将ROS图像消息转换为numpy数组
            # 深度图像通常是32位浮点数或16位无符号整数
            if msg.encoding == '32FC1':
                # 32位浮点深度图
                depth_array = np.frombuffer(msg.data, dtype=np.float32).reshape(
                    msg.height, msg.width)
            elif msg.encoding == '16UC1':
                # 16位无符号整数深度图（通常以毫米为单位）
                depth_array = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                    msg.height, msg.width).astype(np.float32) / 1000.0  # 转换为米
            else:
                rospy.logwarn(f"不支持的深度图像编码: {msg.encoding}")
                return
            # 更新深度图像数据
            self.depth_image = np.array(depth_array).copy()

            self.depth_image_timestamp = msg.header.stamp.to_sec()
            self.depth_image_received = True
            self.depth_image = np.clip(self.downsample_depth_image(self.depth_image).flatten(), 0, 6)


        except Exception as e:
            rospy.logerr(f"深度图像处理错误: {e}")

    def signal_handler(self, sig, frame):
        """处理 Ctrl+C 信号"""
        print("\n收到 Ctrl+C 信号，正在关闭...")
        self.running = False
        self.start_controller = False  # 停止控制器
        rospy.signal_shutdown('用户终止')

    def downsample_depth_image(self, depth_image, target_shape=(11, 18),
                               zero_threshold=0.1, neighborhood_radius=2):
        """
        降采样深度图像，并过滤零值噪声

        参数:
            depth_image: 原始深度图像 (480, 848)
            target_shape: 目标形状 (height, width)
            zero_threshold: 深度值为0的阈值
            neighborhood_radius: 邻域半径，控制取点范围
                            =0: 只取中心点
                            =1: 取3x3邻域（默认）
                            =2: 取5x5邻域
                            =3: 取7x7邻域

        返回:
            downsampled: 降采样后的深度图像 (11, 18)
        """
        # 原始图像形状
        h_orig, w_orig = depth_image.shape
        h_target, w_target = target_shape

        # 计算采样步长
        h_step = h_orig // h_target
        w_step = w_orig // w_target

        # 创建目标数组
        downsampled = np.zeros(target_shape, dtype=depth_image.dtype)

        for i in range(h_target):
            for j in range(w_target):
                # 计算采样中心点
                center_h = i * h_step + h_step // 2
                center_w = j * w_step + w_step // 2

                # 根据邻域半径定义采样区域
                h_start = max(0, center_h - neighborhood_radius)
                h_end = min(h_orig, center_h + neighborhood_radius + 1)
                w_start = max(0, center_w - neighborhood_radius)
                w_end = min(w_orig, center_w + neighborhood_radius + 1)

                # 提取采样区域
                region = depth_image[h_start:h_end, w_start:w_end]

                if depth_image[center_h, center_w] < zero_threshold:
                    downsampled[i, j] = np.mean(region)
                else:
                    downsampled[i, j] = depth_image[center_h, center_w]

        downsampled[:, 0] = downsampled[:, 1]  #byd深度相机左边读不到东西
        return downsampled
