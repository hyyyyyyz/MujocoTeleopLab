"""Check XRoboToolkit body tracking without starting MuJoCo."""

from __future__ import annotations

import argparse
import time

from teleopit.inputs.xrobotoolkit_provider import XRoboToolkitInputProvider


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()

    provider = XRoboToolkitInputProvider(timeout=2.0)
    deadline = time.monotonic() + max(args.seconds, 0.0)
    samples = 0
    try:
        print("Waiting for XRoboToolkit full-body frames. Press Ctrl+C to stop.")
        while time.monotonic() < deadline:
            try:
                _, timestamp_s, seq = provider.get_frame_packet()
            except TimeoutError:
                print("No body frame yet: check PC Service, headset connection, Full-body and Send.")
                continue
            samples += 1
            print(f"frame={samples} seq={seq} timestamp={timestamp_s:.3f} fps={provider.fps:.1f}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        provider.close()


if __name__ == "__main__":
    main()
