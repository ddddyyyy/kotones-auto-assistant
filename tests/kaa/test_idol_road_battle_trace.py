from dataclasses import replace
import pytest
from kaa.idol_road.battle_trace import BattleTrace
from kaa.idol_road.battle_observation import BattleFrameValues, BattleObservationSession, bind_battle_frame
from kaa.idol_road.models import Action, Decision, Observation, Outcome, Scene
from kaa.idol_road.rule_conditions import evaluate_rule_conditions
from kaa.idol_road.rule_catalog import RuleDeclarationCatalog, UnknownRuleSighting


def frame(frame_id='a', hp=44, battle_id='battle', **changes):
    session=BattleObservationSession(battle_id,'ttmr',16,10,'confirmed_formation_start_request')
    return bind_battle_frame(session,replace(BattleFrameValues(frame_id,True,4,hp),**changes))


def observe(trace, value=None, action=Action.WAIT):
    value=value or frame()
    trace.observe(Observation(Scene.BATTLE,'ttmr',16),Decision(action,'request','ttmr',16),value)


def test_adjacent_same_visible_states_compact_but_keep_first_and_last_frame():
    trace=BattleTrace()
    for frame_id in ('a','b','c'):
        observe(trace,frame(frame_id))
    snapshot=trace.snapshot()
    assert len(snapshot.steps)==1 and snapshot.observation_count==3
    step=snapshot.steps[0]
    assert step.battle_frame.values.frame_id=='a' and step.last_frame_id=='c'
    assert step.repeated_observations==3
    assert not snapshot.complete and not snapshot.replay_ready


@pytest.mark.parametrize('next_frame', [frame('b',hp=40),frame('b',hp=None),
    frame('b',battle_id='new'),frame('b',resource_state='selection_preview'),
    frame('b',remaining_turns=3)])
def test_resource_unknown_turn_preview_and_battle_boundaries_not_compacted(next_frame):
    trace=BattleTrace()
    observe(trace)
    observe(trace,next_frame)
    assert len(trace.snapshot().steps)==2


def test_operation_requests_are_not_success_and_different_requests_remain_separate():
    trace=BattleTrace()
    observe(trace,action=Action.INSPECT_BATTLE_HP)
    observe(trace)
    assert [step.requested_decision.action for step in trace.snapshot().steps]==[Action.INSPECT_BATTLE_HP,Action.WAIT]
    assert not trace.snapshot().replay_ready


def test_result_does_not_carry_previous_resources_or_convert_failure_to_zero_stars():
    trace=BattleTrace()
    observe(trace)
    trace.observe(Observation(Scene.RESULT,'ttmr',16,outcome=Outcome.FAILURE),None)
    step=trace.snapshot().steps[-1]
    assert step.scene==Scene.RESULT and step.stars is None
    assert step.battle_frame is None and step.conditions is None
    assert trace.snapshot().steps[0].battle_frame.observation.hp==44


@pytest.mark.parametrize('obs,reason', [(None,'frame_scene_mismatch'),
    (Observation(Scene.RESULT,'ttmr',16),'frame_scene_mismatch'),
    (Observation(Scene.BATTLE,'hski',13),'frame_identity_mismatch')])
def test_stale_frame_cannot_be_attached_to_another_scene_or_identity(obs,reason):
    trace=BattleTrace()
    trace.observe(obs,None,frame())
    step=trace.snapshot().steps[0]
    assert step.battle_frame is None and step.frame_issue==reason


def test_conditions_require_exact_frame_and_battle_and_keep_unknown_sightings():
    trace=BattleTrace()
    current=frame()
    catalog=RuleDeclarationCatalog('ttmr',16,'partial',unknown_sightings=(UnknownRuleSighting(0,1,'unsupported_visible_rule'),))
    report=evaluate_rule_conditions(catalog,current.observation,battle_id='battle')
    observation=Observation(Scene.BATTLE,'ttmr',16)
    trace.observe(observation,None,current,report)
    trace.observe(observation,None,frame('b'),report)
    steps=trace.snapshot().steps
    assert steps[0].conditions.unknown_rule_sightings==catalog.unknown_sightings
    assert not steps[0].conditions.executable
    assert steps[1].conditions is None and steps[1].condition_issue=='report_frame_mismatch'


def test_capacity_eviction_counts_all_compacted_observations_and_snapshot_is_immutable():
    trace=BattleTrace(capacity=2)
    observe(trace)
    observe(trace,frame('b'))
    initial=trace.snapshot()
    observe(trace,frame('c',40))
    observe(trace,frame('d',30))
    snapshot=trace.snapshot()
    assert snapshot.observation_count==4 and snapshot.evicted_observations==2
    assert [s.sequence for s in snapshot.steps]==[3,4]
    assert initial.observation_count==2 and len(initial.steps)==1


@pytest.mark.parametrize('capacity',[0,-1,1025,True,1.5])
def test_capacity_is_explicitly_bounded(capacity):
    with pytest.raises(ValueError):
        BattleTrace(capacity)
