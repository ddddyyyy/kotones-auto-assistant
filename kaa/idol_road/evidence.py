"""可选、有限的爬塔停止证据；只使用任务已经取得的截图。"""

from collections import deque
from dataclasses import asdict
from enum import Enum
import json
import logging
import os
from pathlib import Path
from uuid import uuid4

import cv2
from cv2.typing import MatLike

from .models import Decision, Observation
from .objectives import StageObjectives
from .items import ItemInventory
from .preparation import BattlePreparation
from .rules import StageRules
from .battle_observation import BattleFrameObservation
from .rule_conditions import RuleConditionReport
from .battle_trace import BattleTrace
from .planner_trial import PlannerTrialSpec


logger = logging.getLogger(__name__)
EVIDENCE_ENV = 'KAA_IDOL_ROAD_EVIDENCE_DIR'


def _json_default(value):
    if isinstance(value,Enum):return value.value
    if isinstance(value,(set,frozenset)):return sorted(value,key=str)
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


class RunEvidence:
    def __init__(self, directory: Path) -> None:
        self.directory = directory / f'idol-road-{uuid4().hex}'
        self._frames: deque[MatLike] = deque(maxlen=3)
        self._finished = False
        self._trace = BattleTrace()

    @classmethod
    def from_environment(cls) -> 'RunEvidence | None':
        directory = os.environ.get(EVIDENCE_ENV)
        return cls(Path(directory)) if directory else None

    def observe(self, screen: MatLike) -> None:
        if not self._finished:
            self._frames.append(screen.copy())

    def observe_step(self, observation: Observation | None, decision: Decision | None,
                     battle_frame: BattleFrameObservation | None = None,
                     condition_report: RuleConditionReport | None = None) -> None:
        if not self._finished:
            self._trace.observe(observation, decision, battle_frame, condition_report)

    def finish(
        self, reason: str, observation: Observation | None,
        decision: Decision | None, battles_started: int,
        objectives: StageObjectives | None = None,
        item_inventory: ItemInventory | None = None,
        preparation: BattlePreparation | None = None,
        stage_rules: StageRules | None = None,
        battle_frame: BattleFrameObservation | None = None,
        condition_report: RuleConditionReport | None = None,
        planner_trial: PlannerTrialSpec | None = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            frames = []
            for index, screen in enumerate(self._frames):
                ok, encoded = cv2.imencode('.png', screen)
                if not ok:
                    raise OSError('PNG encoding failed')
                name = f'frame-{index}.png'
                (self.directory / name).write_bytes(encoded.tobytes())
                frames.append(name)
            payload = {
                'version': 1, 'reason': reason, 'battles_started': battles_started,
                'observation': asdict(observation) if observation is not None else None,
                'decision': asdict(decision) if decision is not None else None,
                'objectives': asdict(objectives) if objectives is not None else None,
                'item_inventory': asdict(item_inventory) if item_inventory is not None else None,
                'battle_preparation': asdict(preparation) if preparation is not None else None,
                'stage_rules': asdict(stage_rules) if stage_rules is not None else None,
                'battle_frame': asdict(battle_frame) if battle_frame is not None else None,
                'rule_conditions': asdict(condition_report) if condition_report is not None else None,
                'planner_trial': ({**asdict(planner_trial),'archetype':planner_trial.archetype.value}
                                  if planner_trial is not None else None),
                'battle_trace': 'battle-trace.json',
                'frames': frames,
                'note': 'Frames are observations, not proof of navigation, rewards or battle success.',
            }
            (self.directory / 'stop.json').write_text(
                json.dumps(payload, ensure_ascii=False, indent=2,default=_json_default), encoding='utf-8',
            )
            (self.directory / 'battle-trace.json').write_text(
                json.dumps(asdict(self._trace.snapshot()), ensure_ascii=False, indent=2,default=_json_default), encoding='utf-8',
            )
            logger.info('偶像之路停止证据：%s', self.directory)
        except (OSError, cv2.error,TypeError):
            # 诊断不能遮盖原始停止原因，也不能影响输入决策。
            logger.warning('无法保存偶像之路停止证据。', exc_info=True)
        finally:
            self._frames.clear()
