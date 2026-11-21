"""试跑输入结算包装；场地约束可选，输入记录始终启用。"""
from typing import Callable
from kaa.db.skill_card import SkillCard
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from kaa.tasks.produce.new.play_cards.planner.feedback import PendingPlay,PlayFeedback
from kaa.tasks.produce.new.play_cards.planner.selection_policy import SelectionPolicy,SelectionConstraint,SelectionPolicyStop
from .hp_safety import HpSafetyPolicy
from dataclasses import asdict,replace
import logging
from .battle_observation import BattleFrameObservation
from .battle_inputs import BattleInputs


logger = logging.getLogger(__name__)


class InputBoundPolicy:
    def __init__(self,inputs: BattleInputs,frame: Callable[[],BattleFrameObservation | None],
                 delegate: SelectionPolicy | None=None,guard: HpSafetyPolicy | None=None):
        self.inputs=inputs
        self.frame=frame
        self.delegate=delegate
        self.guard=guard

    def select(self,state: BattleState,cards: tuple[SkillCard,...]) -> SelectionConstraint:
        if not self.inputs.can_select(self.frame(),state):
            if self.guard is not None:
                self.guard.review=None
            return SelectionConstraint(reason='battle_input:selection_state_unbound_or_pending',allowed_indices=(),stop=True)
        if self.guard is None:
            return self.delegate.select(state,cards) if self.delegate is not None else SelectionConstraint()
        safety=self.guard.select(state,cards)
        logger.info('Idol road current-card HP review: %s',
            asdict(self.guard.review) if self.guard.review is not None else None)
        if safety.stop and self.guard.reserve_mode=='prefer' and cards and self.delegate is not None:
            # An unsupported card must not prevent deliberately keeping it.
            # Only ending the turn can use this path; no unfiltered card is
            # authorized, and the independent end-turn guard still applies.
            hold=self.delegate.select(state,cards)
            if hold.end_turn and not hold.stop:
                check=self.guard.end_turn(state)
                if check.stop:
                    return check
                return replace(hold,preferred_indices=(),allowed_indices=(),
                    reason=check.reason+'; '+hold.reason)
        if safety.stop or safety.end_turn:
            return safety
        allowed=safety.allowed_indices
        assert allowed is not None
        subset=tuple(cards[index] for index in allowed)
        choice=self.delegate.select(state,subset) if self.delegate is not None else SelectionConstraint()
        if choice.stop:
            return choice
        if choice.end_turn:
            check=self.guard.end_turn(state)
            if check.stop:
                return check
        indices=choice.allowed_indices if choice.allowed_indices is not None else tuple(range(len(subset)))
        if (any(type(index) is not int or not 0<=index<len(subset) for index in (*indices,*choice.preferred_indices))
            or any(index not in indices for index in choice.preferred_indices)):
            raise SelectionPolicyStop('hp_safety:delegate_candidate_mapping_invalid')
        return replace(choice,preferred_indices=tuple(allowed[index] for index in choice.preferred_indices),
            allowed_indices=tuple(allowed[index] for index in indices),reason=safety.reason+'; '+choice.reason)

    def committed(self,state: BattleState,card: SkillCard) -> None:
        self.inputs.bind_card_state(state,card._id,category=getattr(card,'category',None),
            upgrade_count=getattr(card,'upgrade_count',None))
        if self.delegate is not None:
            self.delegate.committed(state,card)

    def feedback(self,pending: PendingPlay,feedback: PlayFeedback,state: BattleState) -> None:
        frame=self.frame()
        if frame is None or not self.inputs.finish_card(pending,feedback,state,frame):
            raise ValueError('本次爬塔出牌反馈尚未可靠结算，停止后续输入')
        owned=self.inputs.last_owned_card_play
        if owned is not None:
            logger.info('Idol road owned card evidence: %s',asdict(owned))
        if self.delegate is not None:
            self.delegate.feedback(pending,feedback,state)
