"""Seconds <-> SMPTE timecode HH:MM:SS:FF (non-drop frame, 25 fps by default)."""
from __future__ import annotations

import math

TC_FPS = 25


def to_frames(seconds: float, fps: int = TC_FPS) -> int:
    """Frame i covers [i/fps, (i+1)/fps), so a time maps to floor(t * fps). The epsilon stops
    float error (61.04 * 25 = 1525.9999...) from dropping a frame."""
    return int(math.floor(seconds * fps + 1e-6))


def frames_to_tc(frames: int, fps: int = TC_FPS) -> str:
    ff = frames % fps
    s = frames // fps
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}:{ff:02d}"


def to_tc(seconds: float, fps: int = TC_FPS) -> str:
    return frames_to_tc(to_frames(seconds, fps), fps)
