"""有界的任务观察轨迹；记录条件和操作请求，不推断触发、编组或行动成功。"""
from collections import deque
from dataclasses import dataclass, replace
from typing import Literal

from .battle_observation import BattleFrameObservation
from .models import Decision, Observation, Outcome, Scene
from .rule_conditions import RuleConditionReport, RuleConditionResult
from .rule_catalog import UnknownRuleSighting
from .status_details import BattleStatusDetails
from .formation import FormationSlots
from .battle_log import LogInspection,LogReturn
from .battle_inputs import BattleInputState
from .round_anchor import RoundAnchorState
from .skip_probe import SkipLogResolution
from .card_probe import CardLogResolution
from .log_hand import LogHandObservation


@dataclass(frozen=True)
class TraceConditions:
    checks: tuple[RuleConditionResult, ...]
    unknown_rule_sightings: tuple[UnknownRuleSighting, ...]
    context_issue: str | None
    executable: Literal[False] = False


@dataclass(frozen=True)
class BattleTraceStep:
    sequence: int
    scene: Scene
    character_id: str | None
    stage: int | None
    stars: int | None
    outcome: Outcome
    requested_decision: Decision | None
    battle_frame: BattleFrameObservation | None
    conditions: TraceConditions | None
    condition_issue: Literal['report_frame_mismatch'] | None = None
    frame_issue: Literal['frame_scene_mismatch', 'frame_identity_mismatch'] | None = None
    repeated_observations: int = 1
    last_frame_id: str | None = None
    status_details: BattleStatusDetails | None = None
    formation_slots: FormationSlots | None = None
    log_inspection: LogInspection | None = None
    log_return: LogReturn | None = None
    input_state: BattleInputState | None = None
    round_anchor: RoundAnchorState | None = None
    skip_resolution: SkipLogResolution | None = None
    log_hand: LogHandObservation | None = None
    card_resolution: CardLogResolution | None = None


@dataclass(frozen=True)
class BattleTraceSnapshot:
    steps: tuple[BattleTraceStep, ...]
    observation_count: int
    evicted_observations: int
    capacity: int
    version: Literal[1] = 1
    scope: Literal['observations_and_requests_only'] = 'observations_and_requests_only'
    complete: Literal[False] = False
    replay_ready: Literal[False] = False


def _signature(step: BattleTraceStep) -> tuple:
    frame = step.battle_frame
    # OCR证据的帧ID每次不同；只合并相邻且可见状态／条件／操作请求都相同的采样。
    # 保留首帧完整证据和最后帧ID，不将采样间隔推断成连续生效时长。
    state = (frame.session, replace(frame.observation, frame_id=''),
             frame.values.remaining_turns, frame.values.resource_state,frame.values.detail_mode,
             (frame.maximum_hp_parameter.calibration.max_hp,frame.maximum_hp_parameter.calibration.source_frame_id)
             if frame.maximum_hp_parameter is not None else None) if frame is not None else None
    inspection,returned=step.log_inspection,step.log_return
    def header_key(header):
        return (header.status,header.absolute_turn,header.remaining_turns) if header else None
    log=(inspection.request,inspection.viewport.body_fingerprint,inspection.closing_requested,
         inspection.scroll_pending,inspection.scroll_attempts,inspection.expired,inspection.inputs_invalidated,
         inspection.short_list_stable,header_key(inspection.tail_header)) if inspection else None
    back=(returned.request,returned.hud_unchanged,returned.remaining_consistent,returned.input_sequence_consistent,header_key(returned.tail_header)) if returned else None
    anchor=step.round_anchor
    if anchor is not None and anchor.recheck is not None:
        anchor=replace(anchor,recheck=replace(anchor.recheck,frame_id=''))
    return (step.scene, step.character_id, step.stage, step.stars, step.outcome,
            step.requested_decision, state, step.conditions, step.condition_issue, step.frame_issue,step.status_details,
            replace(step.formation_slots,frame_id='') if step.formation_slots else None,log,back,step.input_state,anchor,step.skip_resolution,step.card_resolution,
            (step.log_hand.count,step.log_hand.fingerprint,step.log_hand.empty_confirmed) if step.log_hand else None)


class BattleTrace:
    def __init__(self, capacity: int = 256) -> None:
        if type(capacity) is not int or not 1 <= capacity <= 1024:
            raise ValueError('轨迹容量必须为1到1024的整数')
        self._steps: deque[BattleTraceStep] = deque(maxlen=capacity)
        self._capacity = capacity
        self._observations = 0
        self._evicted = 0

    def observe(self, observation: Observation | None, decision: Decision | None,
                frame: BattleFrameObservation | None = None,
                report: RuleConditionReport | None = None) -> None:
        self._observations += 1
        conditions = None
        issue = None
        frame_issue = None
        if frame is not None:
            if observation is None or observation.scene != Scene.BATTLE:
                frame_issue = 'frame_scene_mismatch'
                frame = None
            elif (observation.character_id, observation.stage) != (frame.session.character_id, frame.session.stage):
                frame_issue = 'frame_identity_mismatch'
                frame = None
        if report is not None:
            if (frame is None or report.observation != frame.observation
                or report.expected_battle_id != frame.session.battle_id):
                issue = 'report_frame_mismatch'
            else:
                conditions = TraceConditions(report.checks, report.unknown_rule_sightings, report.context_issue)
        step = BattleTraceStep(
            self._observations, observation.scene if observation is not None else Scene.UNKNOWN,
            observation.character_id if observation is not None else None,
            observation.stage if observation is not None else None,
            observation.stars if observation is not None else None,
            observation.outcome if observation is not None else Outcome.UNKNOWN,
            decision, frame, conditions, issue, frame_issue,
            last_frame_id=frame.values.frame_id if frame is not None else None,
            status_details=(observation.battle_status_details if observation is not None
                            and observation.scene==Scene.BATTLE_STATUS_DETAILS else None),
            formation_slots=(observation.formation_slots if observation is not None and observation.scene==Scene.CONFIRM else None),
            log_inspection=(observation.battle_log_inspection if observation is not None and observation.scene==Scene.BATTLE_LOG else None),
            log_return=(observation.battle_log_return if observation is not None and observation.scene==Scene.BATTLE else None),
            input_state=(observation.battle_input_state if observation is not None and observation.scene==Scene.BATTLE else None),
            round_anchor=(observation.battle_round_anchor if observation is not None and observation.scene==Scene.BATTLE else None),
            skip_resolution=(observation.battle_skip_resolution if observation is not None and observation.scene==Scene.BATTLE else None),
            card_resolution=(observation.battle_card_resolution if observation is not None and observation.scene==Scene.BATTLE else None),
            log_hand=(observation.battle_log_hand if observation is not None and observation.scene==Scene.BATTLE else None),
        )
        if self._steps and _signature(self._steps[-1]) == _signature(step):
            previous = self._steps[-1]
            self._steps[-1] = replace(previous, repeated_observations=previous.repeated_observations + 1,
                                      last_frame_id=step.last_frame_id)
            return
        if len(self._steps) == self._capacity:
            self._evicted += self._steps[0].repeated_observations
        self._steps.append(step)

    def snapshot(self) -> BattleTraceSnapshot:
        return BattleTraceSnapshot(tuple(self._steps), self._observations, self._evicted, self._capacity)
