"""Load the LinkerHand calibration shared with the host policy service."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib.resources import files


@dataclass(frozen=True)
class HandCalibration:
    open_raw: tuple[float, ...]
    close_raw: tuple[float, ...]
    range_tolerance: float

    @classmethod
    def load(cls) -> "HandCalibration":
        path = files("teleopit.high_level_policy").joinpath("hand_calibration.json")
        try:
            document = json.loads(path.read_bytes())
            opened = tuple(float(value) for value in document["open_raw"])
            closed = tuple(float(value) for value in document["close_raw"])
            range_tolerance = float(document["range_tolerance"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid hand_calibration.json: {exc}") from exc

        if len(opened) != 6 or len(closed) != 6:
            raise ValueError("hand_calibration.json must define six open_raw and close_raw values")
        if not all(math.isfinite(value) for value in (*opened, *closed, range_tolerance)):
            raise ValueError("hand_calibration.json values must be finite")
        if any(opened_value == closed_value for opened_value, closed_value in zip(opened, closed, strict=True)):
            raise ValueError("Each hand_calibration.json open/close pair must differ")

        return cls(
            open_raw=opened,
            close_raw=closed,
            range_tolerance=range_tolerance,
        )
