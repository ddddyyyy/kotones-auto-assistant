"""Absolute-round consumer rules with synthetic observations; no live-turn claims."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any,cast
import pytest
from kaa.db.constants import ProduceExamEffectType
from kaa.tasks.produce.new.play_cards.planner.memory import BattleMemory
from kaa.tasks.produce.new.play_cards.planner.state import AnomalyState,AnomalyStance,ExamBuffState,BattleState
from kaa.tasks.produce.new.play_cards.planner.turn_observation import AbsoluteTurnObservation
from kaa.tasks.produce.new.play_cards.planner.strategy import PlannerStrategy
from kaa.tasks.produce.new.play_cards.planner.feedback import PendingPlay,evaluate_play_feedback,PlayFeedbackStatus
from kaa.idol_road.turn_end import UnplayableTurnGate,PlannedTurnWait


def clock(turn=16,remaining=5,frame='frame',battle='battle'):
    return AbsoluteTurnObservation(battle,frame,turn,remaining)


def card(name):
    return cast(Any,SimpleNamespace(_id=name))


def memory():
    return BattleMemory(buffs=ExamBuffState(parameter_buff_turns=7,parameter_buff_multiple_turns=4),
        anomaly=AnomalyState(stance=AnomalyStance.PRESERVATION,full_power_points=10))


def test_first_anchor_does_not_invent_sixteen_elapsed_rounds():
    state=memory()
    state.observe(5,[card('a')],absolute_turn=clock())
    assert state.buffs.parameter_buff_turns==7
    assert state.anomaly.stance==AnomalyStance.PRESERVATION
    assert state.discard==[]


@pytest.mark.parametrize('remaining',[4,5,6])
def test_actual_new_round_advances_even_with_unchanged_or_increased_remaining(remaining):
    state=memory()
    state.observe(5,[card('a'),card('b')],absolute_turn=clock())
    state.plays_this_turn=3
    state.observe(remaining,[card('c')],absolute_turn=clock(17,remaining,'next'))
    assert state.plays_this_turn==0 and state.discard==['a','b']
    assert state.buffs.parameter_buff_turns==6 and state.buffs.parameter_buff_multiple_turns==3
    assert state.anomaly.full_power_change_count==1 and state.anomaly.full_power_active
    assert state.anomaly.enthusiasm==5 and state.anomaly.extra_play_count==2
    assert state.anomaly.stance==AnomalyStance.NORMAL


@pytest.mark.parametrize('remaining',[2,5,6])
def test_horizon_change_on_same_round_does_not_decay_or_reset(remaining):
    state=memory()
    state.observe(5,[card('a')],absolute_turn=clock())
    state.plays_this_turn=2
    state.observe(remaining,[card('a')],absolute_turn=clock(16,remaining,'new'))
    assert state.plays_this_turn==2 and state.discard==[]
    assert state.buffs.parameter_buff_turns==7 and not state.anomaly.full_power_active


def test_duplicate_observation_cannot_apply_turn_transition_twice():
    state=memory()
    state.observe(5,[card('a')],absolute_turn=clock())
    state.observe(5,[card('b')],absolute_turn=clock(17,5,'new'))
    state.plays_this_turn=1
    state.observe(5,[card('b')],absolute_turn=clock(17,5,'new'))
    assert state.plays_this_turn==1 and state.discard==['a']
    assert state.buffs.parameter_buff_turns==6 and state.anomaly.full_power_change_count==1


def test_jump_decays_known_rounds_and_discards_only_the_last_seen_hand():
    state=memory()
    state.observe(5,[card('a')],absolute_turn=clock())
    state.observe(3,[card('b')],absolute_turn=clock(19,3,'new'))
    assert state.discard==['a'] and state.buffs.parameter_buff_turns==4
    assert set(state.known_cards)=={'a','b'}


@pytest.mark.parametrize('observation',[
    clock(15,5,'next'),clock(17,5,'next','other'),clock(17,5,'frame'),
    clock(True),clock(0),clock(100),clock(frame=''),clock(battle=''),clock(remaining=0),
    clock(remaining=True),clock(remaining=31),None])
def test_invalid_stale_or_missing_anchor_rejected_before_any_memory_mutation(observation):
    state=memory()
    state.observe(5,[card('a')],absolute_turn=clock())
    before=deepcopy(state)
    with pytest.raises(ValueError):
        state.observe(5,[card('unknown')],absolute_turn=observation)
    assert state==before


def test_mismatched_horizon_rejected_before_first_observation():
    state=memory()
    before=deepcopy(state)
    with pytest.raises(ValueError):
        state.observe(4,[card('a')],absolute_turn=clock())
    assert state==before


def test_legacy_mode_stays_compatible_and_cannot_switch_mid_battle():
    state=memory()
    state.observe(5,[card('a')])
    state.observe(3,[card('b')])
    assert state.buffs.parameter_buff_turns==5 and state.discard==['a']
    before=deepcopy(state)
    with pytest.raises(ValueError):
        state.observe(3,[card('c')],absolute_turn=clock(18,3,'new'))
    assert state==before


class Context:
    def __init__(self,observation):
        self.observation=observation

    def fetch_absolute_turn_observation(self):
        return self.observation

    def fetch_remaining_turns(self):
        raise AssertionError('Absolute path must not use generic remaining OCR')

    def fetch_hp(self):
        return 40

    def fetch_stamina(self):
        return 0


def test_strategy_uses_bound_horizon_and_actual_clock_then_preserves_identity_in_state():
    planner=PlannerStrategy(archetype=ProduceExamEffectType.ExamParameterBuff)
    context=Context(clock())
    first=planner._build_observed_state(cast(Any,context),[])
    planner._memory.plays_this_turn=2
    planner._memory.buffs=ExamBuffState(parameter_buff_turns=7)
    context.observation=clock(17,5,'next')
    second=planner._build_observed_state(cast(Any,context),[])
    assert first.absolute_turn_observation==clock() and second.absolute_turn_observation==context.observation
    assert second.remaining_turns==5 and second.remaining_turns_known
    assert planner._memory.plays_this_turn==0 and second.buffs.parameter_buff_turns==6
    before=deepcopy(planner._memory)
    context.observation=None
    with pytest.raises(ValueError):
        planner._build_observed_state(cast(Any,context),[])
    assert planner._memory==before


def snapshot(observation):
    return BattleState(observation.remaining_turns,40,0,hand_card_ids=('played','other'),hand_size=2,
                       absolute_turn_observation=observation)


def test_feedback_does_not_confuse_added_round_with_same_round_card_success():
    before=snapshot(clock())
    after=replace(snapshot(clock(17,5,'next')),hand_card_ids=('other',),hand_size=1,hp=35)
    feedback=evaluate_play_feedback(PendingPlay('played',before,0,0),after)
    assert feedback.status==PlayFeedbackStatus.UNCONFIRMED
    assert feedback.evidence=='turn_advanced_without_play_observation'


@pytest.mark.parametrize('change',[
    {'absolute_turn_observation':None},
    {'absolute_turn_observation':clock(15,5,'next')},
    {'absolute_turn_observation':clock(16,5,'next','other')},
    {'absolute_turn_observation':clock(16,4,'next')},
    {'projected_effect_values':()},
])
def test_feedback_unknown_or_conflicting_round_never_uses_remaining_fallback(change):
    before=snapshot(clock())
    after=replace(before,hand_card_ids=('other',),hand_size=1,hp=35,**change)
    assert evaluate_play_feedback(PendingPlay('played',before,0,0),after).status==PlayFeedbackStatus.UNCONFIRMED


def test_actual_same_round_can_confirm_card_even_when_horizon_changes():
    before=snapshot(clock())
    after=replace(snapshot(clock(16,6,'next')),hand_card_ids=('other',),hand_size=1)
    assert evaluate_play_feedback(PendingPlay('played',before,0,0),after).evidence=='card_left_hand'


def test_skip_gate_rejects_round_regression_and_another_battle_without_clearing_pending():
    gate=UnplayableTurnGate()
    assert gate.step('battle',16,True,True)=='click'
    for battle,turn in (('battle',15),('other',17)):
        assert gate.step(battle,turn,True,True)=='stop'
        assert gate.pending==('battle',16)
    assert gate.step('battle',17,False,True)=='none'
    assert gate.step('battle',16,True,True)=='stop'


def test_repeated_planned_skip_request_cannot_renew_timeout():
    now=[0.]
    gate=PlannedTurnWait(clock=lambda:now[0])
    gate.requested(16)
    now[0]=19
    with pytest.raises(ValueError):
        gate.requested(16)
    now[0]=20
    assert gate.check(16)=='stop'
