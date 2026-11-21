from dataclasses import replace
import pytest
from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame,create_battle_session
from kaa.idol_road.rules import RuleTextRead,RuleTextLine,StageRules,RuleViewport,VisibleStageRule
from kaa.idol_road.rule_symbols import RuleSymbolEvidence
from kaa.idol_road.models import Observation,Scene
from kaa.idol_road.page import PageSnapshot
from kaa.idol_road.planner_adapter import TrialFrameGuard
from kaa.idol_road.planner_context import bind_concentration


def rules():
    title=tuple(RuleTextRead('frame',kind,(RuleTextLine('16ターン目',.99),)) for kind in ('original','grayscale'))
    body=tuple(replace(read,lines=(RuleTextLine('body unknown',.8),)) for read in title)
    row=VisibleStageRule((40,640,637,185),16,title,body,
        turn_addition=RuleSymbolEvidence('frame','turn_addition',(385,749,225,43),.99))
    return StageRules('ttmr',20,(RuleViewport('page',(row,)),))


def variable():
    formation=Observation(Scene.CONFIRM,'ttmr',20,stars=0,formation_ready=True,total_turns=20)
    return create_battle_session(formation,stage_rules=rules())


@pytest.mark.parametrize('remaining',[20,5,4,1,21,None])
def test_dynamic_budget_never_fabricates_absolute_turn_from_initial_total(remaining):
    session=variable()
    assert session.turn_budget_mode=='variable_budget' and session.initial_total_turns==20
    frame=bind_battle_frame(session,BattleFrameValues('hud',True,remaining,48))
    assert frame is not None and frame.observation.turn is None and frame.observation.turn_source is None
    assert frame.observation.hp==48 and frame.values.remaining_turns==remaining
    assert 'turn' not in frame.observation.conflicted_fields
    assert session.total_turns==20


def test_same_remaining_value_cannot_distinguish_extended_rounds():
    session=variable()
    for identifier in ('before-extra','after-extra','new-screenshot'):
        frame=bind_battle_frame(session,BattleFrameValues(identifier,True,5,48))
        assert frame is not None and frame.observation.turn is None


def test_fixed_budget_compatibility_is_explicitly_labelled_as_assumption():
    session=BattleObservationSession('battle','ttmr',19,20,'confirmed_formation_start_request')
    frame=bind_battle_frame(session,BattleFrameValues('frame',True,5,48))
    assert frame is not None and frame.observation.turn==16
    assert frame.observation.turn_source=='fixed_budget_assumption'
    assert session.initial_total_turns==20


@pytest.mark.parametrize('changes',[{'stage':19},{'identity_source':'history'},{'source':'database'}])
def test_creation_rejects_unrelated_rule_metadata(changes):
    formation=Observation(Scene.CONFIRM,'ttmr',20,stars=0,formation_ready=True,total_turns=20)
    with pytest.raises(ValueError,match='规则来源'):
        create_battle_session(formation,stage_rules=replace(rules(),**changes))


def test_dynamic_unknown_frame_cannot_authorize_planner_or_repeated_skip():
    session=variable()
    frame=bind_battle_frame(session,BattleFrameValues('frame',True,5,48))
    assert frame is not None
    with pytest.raises(ValueError,match='可靠当前回合'):
        bind_concentration([],frame)
    now=[0.]
    guard=TrialFrameGuard(session,100,clock=lambda:now[0])
    snapshot=PageSnapshot(Observation(Scene.BATTLE,'ttmr',20),battle_frame=frame)
    assert guard.check(snapshot)=='wait'
    now[0]=21
    with pytest.raises(ValueError,match='持续未知'):
        guard.check(snapshot)


def test_preview_never_exposes_a_derived_absolute_turn():
    session=BattleObservationSession('battle','ttmr',19,20,'confirmed_formation_start_request')
    frame=bind_battle_frame(session,BattleFrameValues('frame',True,5,48,resource_state='selection_preview'))
    assert frame is not None and frame.observation.turn is None and frame.observation.turn_source is None
