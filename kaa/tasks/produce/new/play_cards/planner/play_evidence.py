"""Opt-in evidence for paid plays with no observable state change."""

import json
import os
from pathlib import Path
from uuid import uuid4

import cv2
from cv2.typing import MatLike
from kotonebot import device, logging

logger = logging.getLogger(__name__)
EVIDENCE_ENV = 'KAA_PLAY_EVIDENCE_DIR'


class PlayFeedbackEvidence:
    """Keep one pending play in memory and at most eight anomalous bundles.

    Capturing immediately after the existing double click adds screenshot
    latency in diagnostic mode only; it does not add clicks or retries.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self.saved_count = 0
        self.pending: tuple[str, str, MatLike, MatLike | None] | None = None

    @classmethod
    def from_environment(cls) -> 'PlayFeedbackEvidence | None':
        directory = os.environ.get(EVIDENCE_ENV)
        return cls(Path(directory)) if directory else None

    def before_commit(self, card_id: str, screen: MatLike) -> None:
        self.pending = None
        if self.saved_count < 8:
            self.pending = (uuid4().hex, card_id, screen.copy(), None)

    def after_commit(self) -> None:
        if self.pending is None:
            return
        try:
            token, card_id, before, _ = self.pending
            # Context fetch_screen is memoized and could return a pre-click
            # HUD. This diagnostic must use a fresh device frame instead.
            self.pending = (token, card_id, before, device.screenshot().copy())
        except Exception:
            logger.warning('Could not capture post-click play evidence.', exc_info=True)

    def feedback(self, evidence: str, screen: MatLike | None) -> None:
        pending, self.pending = self.pending, None
        if pending is None or evidence != 'paid_play_observably_unchanged':
            return
        self.saved_count += 1
        token, card_id, before, after = pending
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            prefix = self.directory / f'play-{token}'
            for name, frame in [('before', before), ('after-click', after), ('feedback', screen)]:
                if frame is None:
                    continue
                ok, encoded = cv2.imencode('.png', frame)
                if not ok:
                    raise OSError('PNG encoding failed')
                Path(f'{prefix}-{name}.png').write_bytes(encoded.tobytes())
            Path(f'{prefix}.json').write_text(
                json.dumps({'card_id': card_id, 'evidence': evidence}, ensure_ascii=False),
                encoding='utf-8',
            )
            logger.info('Planner unchanged-play evidence: card=%s prefix=%s', card_id, prefix)
        except (OSError, cv2.error):
            logger.warning('Could not save play feedback evidence.', exc_info=True)
