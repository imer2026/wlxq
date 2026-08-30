"""DebugRecorder 截图落盘测试。"""

from __future__ import annotations

import cv2
import numpy as np

from wlxq_bot.debug.recorder import DebugRecorder
from wlxq_bot.models import MatchResult


def test_save_duplicate_skill_annotated_frame(tmp_path) -> None:
    recorder = DebugRecorder(str(tmp_path))
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    matches = [
        MatchResult("skills/assault/a.png", (50, 60), 0.82),
        MatchResult("skills/assault/a.png", (150, 60), 0.94),
    ]

    path = recorder.save_annotated(
        frame,
        matches,
        frame_id=7,
        prefix="skill_duplicate_assault/a",
    )

    assert path.is_file()
    assert path.parent == tmp_path
    assert "/" not in path.name
    saved = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert saved is not None
    assert saved.shape == frame.shape
