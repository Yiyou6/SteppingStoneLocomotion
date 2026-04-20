from isaaclab.app import AppLauncher
import argparse

def App_Setup(device,headless):
    # add argparse arguments
    parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    # parse the arguments
    args_cli = parser.parse_args()
    args_cli.device = device  # set the device to cuda:0
    args_cli.headless = headless  # uncomment this to run in headless mode

    args_cli.enable_cameras = True  # uncomment this to enable camera rendering
    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app