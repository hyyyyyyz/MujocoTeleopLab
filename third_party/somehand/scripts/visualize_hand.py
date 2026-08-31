"""Standalone hand model viewer."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from somehand.infrastructure.hand_model import HandModel
from somehand.visualization import HandVisualizer


def main():
    parser = argparse.ArgumentParser(description="View a hand MJCF model in MuJoCo")
    parser.add_argument(
        "--mjcf",
        default="assets/mjcf/linkerhand_l20_right/model.xml",
        help="Path to MJCF model file",
    )
    args = parser.parse_args()

    hand_model = HandModel(args.mjcf)
    print(f"Model loaded: nq={hand_model.nq}, nv={hand_model.nv}, nu={hand_model.nu}")
    print(f"Joints: {hand_model.get_joint_names()}")
    print(f"Sites: {hand_model.get_site_names()}")
    print(f"Bodies: {hand_model.get_body_names()}")

    visualizer = HandVisualizer(hand_model)
    print("Viewer launched. Close the window to exit.")

    while visualizer.is_running:
        visualizer.update(hand_model.get_qpos())
        time.sleep(0.01)


if __name__ == "__main__":
    main()
