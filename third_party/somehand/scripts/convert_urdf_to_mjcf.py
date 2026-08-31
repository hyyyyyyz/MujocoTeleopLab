"""CLI tool to convert URDF hand models to MJCF."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from somehand.urdf_converter import convert_urdf_to_mjcf


def main():
    parser = argparse.ArgumentParser(description="Convert URDF to MJCF for somehand")
    parser.add_argument("--urdf", required=True, help="Path to URDF file")
    parser.add_argument("--output", required=True, help="Output directory for MJCF")
    parser.add_argument("--name", default=None, help="Model name (default: URDF stem)")
    args = parser.parse_args()

    convert_urdf_to_mjcf(args.urdf, args.output, args.name)


if __name__ == "__main__":
    main()
