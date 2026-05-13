import numpy as np
import torch
from typing import Tuple

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, ImuCfg, patterns, RayCasterCameraCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfSteppingStonesTerrainCfg, HfRandomUniformTerrainCfg
from isaaclab.sim import DomeLightCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def create_environment(file_path: str,
                       dt: float,
                       sub_step: int,
                       agents_num: int,
                       device: str,
                       domain_randomization_cfg) -> Tuple[sim_utils.SimulationContext, InteractiveScene]:
    """
    创建 Isaac Lab 仿真环境，包括地形、机器人、传感器以及可选的域随机化。

    Args:
        file_path: 机器人 USD 文件路径。
        dt: 环境主时间步长（秒）。
        sub_step: 每个主时间步内的子步数。
        agents_num: 并行环境（智能体）数量。
        device: 计算设备（如 "cuda:0"）。
        domain_randomization_cfg: 包含域随机化参数的对象，
                                  应包含 mass_range, com_range, inertia_range,
                                  friction_range, restitution_range 属性。

    Returns:
        sim: 仿真上下文实例。
        scene: 交互场景实例。
    """
    # 1. 初始化仿真上下文
    sim = _create_simulation_context(dt, sub_step, device)

    # 2. 构建场景配置并实例化场景
    scene_cfg = _build_scene_config(file_path, agents_num)
    scene = InteractiveScene(scene_cfg)

    # 3. 重置仿真和场景到初始状态
    sim.reset()
    scene.reset()
    scene.update(dt=0)

    # 4. 执行域随机化（如果配置了范围）
    _apply_domain_randomization(scene, domain_randomization_cfg, agents_num, device)


    return sim, scene


def _create_simulation_context(dt: float, sub_step: int, device: str) -> sim_utils.SimulationContext:
    """创建并返回仿真上下文实例。"""
    sim_cfg = sim_utils.SimulationCfg(dt=dt / sub_step, device=device)
    return sim_utils.SimulationContext(sim_cfg)


def _build_terrain_generator_config(num_rows: int = 5) -> TerrainGeneratorCfg:
    """
    构建地形生成器配置，包含多个垫脚石地形，用于训练。

    Args:
        num_rows: 地形网格的行数（和列数）。

    Returns:
        地形生成器配置对象。
    """
    terrain_number = 3
    sub_terrains = {}

    # 生成多个不同深度的垫脚石地形
    depths = 0*np.linspace(-0.15, -0.2, terrain_number)
    for i, depth in enumerate(depths):
        sub_terrains[f"stepping_stone{i}"] = HfSteppingStonesTerrainCfg(
            proportion=1.0,
            border_width=0.1,
            holes_depth=float(depth),          # 负值表示下陷深度
            stone_height_max=0.0,
            stone_width_range=(0.5, 0.8),
            stone_distance_range=(0.05, 0.2),
            platform_width=0.7,
        )

    return TerrainGeneratorCfg(
        num_rows=num_rows,
        num_cols=num_rows,
        size=(10.0, 4.0),                      # 每个子地形的尺寸 (m)
        color_scheme="none",
        sub_terrains=sub_terrains,
        curriculum=False,
        border_width=10.0,
        horizontal_scale=0.05,                  # 高分辨率网格
    )


@configclass
class _RobotSceneCfg(InteractiveSceneCfg):
    """
    机器人仿真场景配置类（内部使用）。
    包含光照、地形、机器人本体以及各种传感器。
    """
    # 光照
    # light = AssetBaseCfg(
    #     prim_path="/World/skyLight",
    #     spawn=DomeLightCfg(
    #         intensity=750.0,
    #         color=(0.9, 0.9, 0.9),
    #         texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    #     ),
    # )

    # 地形导入器（使用生成器）
    terrain = TerrainImporterCfg(
        prim_path="/World/defaultGroundPlane",
        terrain_type="generator",
        terrain_generator=_build_terrain_generator_config(),
        debug_vis=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # 机器人本体
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=None,                      # 将在 _build_scene_config 中动态赋值
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        actuators={
            "wheel_acts": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                damping=None,
                stiffness=None,
            )
        },
        collision_group=0,
    )

    # 左脚碰撞传感器
    L_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ankle_L_Link",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )
    # 右脚碰撞传感器
    R_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ankle_R_Link",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )

    # 机身位姿IMU
    imu_sensor = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_Link",
        update_period=0.0,
    )
    # 左脚位姿IMU
    L_imu_sensor = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ankle_L_Link",
        update_period=0.0,
    )
    # 右脚位姿IMU
    R_imu_sensor = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ankle_R_Link",
        update_period=0.0,
    )

    # 深度相机（射线投射式，非渲染式）
    Depth_Camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Camera_Frame",
        update_period=0.0,
        mesh_prim_paths=["/World/defaultGroundPlane"],
        max_distance=6.0,
        depth_clipping_behavior="max",
        debug_vis=False,
        offset=RayCasterCameraCfg.OffsetCfg(convention="world"),
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            width=int(11 * (45.55 / 26.6)),
            height=11,
            horizontal_aperture=45.55,
            vertical_aperture=26.6,
        ),
    )


def _build_scene_config(file_path: str, agents_num: int) -> _RobotSceneCfg:
    """
    构建场景配置对象，并将机器人 USD 路径注入。

    Args:
        file_path: 机器人 USD 文件路径。
        agents_num: 并行环境数量。

    Returns:
        配置好的场景配置对象。
    """
    # 创建配置实例
    cfg = _RobotSceneCfg(num_envs=agents_num, env_spacing=0.0)

    # 动态设置机器人的 USD 路径（避免在类定义中固定）
    cfg.robot.spawn.usd_path = file_path

    return cfg


def _apply_domain_randomization(scene: InteractiveScene, dr_cfg, agents_num: int, device: str) -> None:
    """
    对场景中的机器人应用域随机化：质量、质心、惯性、摩擦力、恢复系数。

    Args:
        scene: 交互场景实例。
        dr_cfg: 域随机化配置对象，应包含以下属性（均为 float 或 Tuple[float, float]）：
                mass_range, com_range, inertia_range, friction_range, restitution_range。
        agents_num: 机器人数量。
        device: 计算设备。
    """
    # 提取各随机化范围
    mass_range = dr_cfg.mass_range
    com_range = dr_cfg.com_range
    inertia_range = dr_cfg.inertia_range
    friction_range = dr_cfg.friction_range
    restitution_range = dr_cfg.restitution_range

    # 获取所有机器人的根物理视图
    """ 域随机化"""
    rb_view = scene["robot"].root_physx_view
    indices = torch.arange(agents_num)
    """ 质量随机化"""
    origin_mass = rb_view.get_masses().clone()
    origin_mass += mass_range * origin_mass * (2 * torch.rand_like(origin_mass) - 1)
    rb_view.set_masses(origin_mass.clamp(0.001, 10000), indices)

    """ 质心随机化"""  # 注意，com不仅包含xyz，还包含四元数
    origin_com = rb_view.get_coms().clone()
    origin_com[..., :3] += com_range * origin_com[..., :3] * (2 * torch.rand_like(origin_com[..., :3]) - 1)
    rb_view.set_coms(origin_com, indices)

    """ 惯性张量随机化"""
    origin_inertia = rb_view.get_inertias().clone()
    origin_inertia += inertia_range * origin_inertia * (2 * torch.rand_like(origin_inertia[..., -1]).unsqueeze(-1)
                                                        - 1.0)

    rb_view.set_inertias(origin_inertia, indices)

    """ 摩擦力与弹力随机化"""
    origin_material = rb_view.get_material_properties().clone()

    origin_material[..., 0:2] += (friction_range * torch.rand_like(origin_material[..., 0:2]))  # only add friction
    origin_material[..., 0:2].clamp_(0.5, 2.0)
    origin_material[..., 2] += (restitution_range * torch.rand_like(origin_material[..., 2]))
    origin_material[..., 2].clamp_(0.0, 1)  # restitution max=1
    rb_view.set_material_properties(origin_material, indices)