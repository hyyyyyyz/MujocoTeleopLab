"""Compatibility re-exports for runtime output sinks."""

from somehand.runtime.sink_outputs import (
    AsyncBiHandLandmarkOutputSink,
    AsyncLandmarkOutputSink,
    BiHandOutputWindowSink,
    BiHandVideoOutputSink,
    RobotHandOutputSink,
    RobotHandTargetOutputSink,
    RobotHandVideoOutputSink,
    TrajectoryRecorder,
)
from somehand.runtime.sink_rendering import (
    create_offscreen_renderer as _create_offscreen_renderer,
    fit_video_size as _fit_video_size,
    transform_points as _transform_points,
)

__all__ = [
    "AsyncBiHandLandmarkOutputSink",
    "AsyncLandmarkOutputSink",
    "BiHandOutputWindowSink",
    "BiHandVideoOutputSink",
    "RobotHandOutputSink",
    "RobotHandTargetOutputSink",
    "RobotHandVideoOutputSink",
    "TrajectoryRecorder",
    "_create_offscreen_renderer",
    "_fit_video_size",
    "_transform_points",
]
