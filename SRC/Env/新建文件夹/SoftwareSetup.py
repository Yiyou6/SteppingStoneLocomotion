from isaaclab.app import AppLauncher
import argparse

def App_Setup(device,headless):
    # 1. 创建参数解析器
    parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")

    # 2. 添加 AppLauncher 所需的命令行参数
    AppLauncher.add_app_launcher_args(parser)

    # 3. 解析命令行参数
    args = parser.parse_args()

    # 4. 根据运行环境覆盖或补充关键参数（这些变量应在外部定义）
    args.device = device  # 强制使用 GPU:0（原 device 变量应已定义）
    args.headless = headless  # 是否无头模式运行（原 headless 变量应已定义）
    args.enable_cameras = True  # 启用相机渲染

    # 5. 使用最终参数启动 Omniverse 应用
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app