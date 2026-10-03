"""Opt-in, bounded frame capture for auditing transient lesson results."""

from collections import deque
import os
from pathlib import Path
from uuid import uuid4

import cv2
from cv2.typing import MatLike
from kotonebot import logging

logger = logging.getLogger(__name__)
EVIDENCE_ENV = 'KAA_PRODUCE_EVIDENCE_DIR'


class PracticeResultEvidence:
    def __init__(self, directory: Path):
        self.directory = directory
        self.frames: deque[MatLike] = deque(maxlen=6)
        self.token = uuid4().hex
        self.finished = False

    @classmethod
    def from_environment(cls) -> 'PracticeResultEvidence | None':
        directory = os.environ.get(EVIDENCE_ENV)
        return cls(Path(directory)) if directory else None

    def observe(self, screen: MatLike, result: str | None) -> None:
        if self.finished:
            return
        self.frames.append(screen.copy())
        if result is not None:
            self._save(self.frames[-1], result, 0)
            self.frames.clear()
            self.finished = True

    def finish(self) -> None:
        if self.finished:
            return
        # An unrecognized outcome stays unconfirmed: these frames are evidence
        # for manual review, never an inferred grade or a reason to click.
        for index, screen in enumerate(self.frames):
            self._save(screen, 'unconfirmed', index)
        self.frames.clear()
        self.finished = True

    def _save(self, screen: MatLike, result: str, index: int) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f'lesson-{self.token}-{result}-{index}.png'
            ok, encoded = cv2.imencode('.png', screen)
            if not ok:
                raise OSError('PNG encoding failed')
            path.write_bytes(encoded.tobytes())
            logger.info('Practice result evidence: result=%s path=%s', result, path)
        except (OSError, cv2.error):
            logger.warning('Could not save practice result evidence.', exc_info=True)
