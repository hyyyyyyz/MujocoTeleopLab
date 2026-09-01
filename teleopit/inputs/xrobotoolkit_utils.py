"""Small safety helpers shared by the XRoboToolkit input processes.

The released XRoboToolkit Python binding has been observed to block forever in
``close()`` when the PC service or headset is already disconnected.  Shutdown
paths must therefore never call that method directly from the main thread.
This module intentionally has no NumPy or SDK imports so it can also be used by
the lightweight Python 3.12 controller bridge.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any


DEFAULT_SDK_CLOSE_TIMEOUT_S = 2.0


def close_sdk_bounded(
    sdk: Any,
    *,
    timeout_s: float = DEFAULT_SDK_CLOSE_TIMEOUT_S,
    logger: logging.Logger | None = None,
    context: str = "XRoboToolkit SDK",
) -> bool:
    """Invoke an SDK ``close`` method without allowing shutdown to hang.

    Some XRoboToolkit releases wait on a feedback-stream worker forever when
    the service has gone away.  The close call is isolated in a daemon thread;
    the caller waits only up to ``timeout_s`` and then continues.  A daemon
    thread is deliberate here: Python cannot safely interrupt a foreign SDK
    call, but it also must not keep a launcher/process alive during teardown.

    Returns ``True`` when the method was absent or returned before the
    deadline, and ``False`` when it timed out or raised.  Exceptions are
    logged and suppressed because this helper is used from cleanup handlers.
    ``timeout_s`` values that are non-finite or negative are treated as zero,
    which makes a malformed config fail safe (immediate return).
    """

    close = getattr(sdk, "close", None)
    if not callable(close):
        return True

    try:
        timeout = float(timeout_s)
    except (TypeError, ValueError, OverflowError):
        timeout = 0.0
    if not math.isfinite(timeout) or timeout < 0.0:
        timeout = 0.0

    log = logger if logger is not None else logging.getLogger(__name__)
    finished = threading.Event()
    error: list[BaseException] = []

    def invoke() -> None:
        try:
            close()
        except BaseException as exc:  # foreign SDKs may raise non-Exception types
            error.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(
        target=invoke,
        name="xrobotoolkit_sdk_close",
        daemon=True,
    )
    try:
        worker.start()
    except BaseException as exc:
        # Thread creation can fail during interpreter/resource teardown (for
        # example, after the process has exhausted its thread quota).  Cleanup
        # callers must never lose the original shutdown exception or leave a
        # misleading unhandled error on this best-effort path.
        log.warning("%s close() worker could not start: %s", context, exc)
        return False
    if not finished.wait(timeout):
        log.warning(
            "%s close() did not return within %.2fs; continuing shutdown",
            context,
            timeout,
        )
        return False

    # The event guarantees that the target has run its finally block.  A
    # zero-time join releases the finished thread's bookkeeping without ever
    # reintroducing an unbounded wait.
    worker.join(timeout=0.0)
    if error:
        log.warning("%s close() raised: %s", context, error[0])
        return False
    return True
