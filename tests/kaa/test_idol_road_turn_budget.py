from dataclasses import replace
import pytest
from kaa.config.idol_road import IdolRoadConfig
from kaa.idol_road.turn_budget import fixed_turn_budget_conflicts
from kaa.idol_road.rule_symbols import RuleSymbolEvidence
from kaa.idol_road.rules import RuleTextRead,RuleTextLine,VisibleStageRule,RuleViewport,StageRules
from kaa.idol_road.planner_trial import prepare_planner_trial,UnsupportedTurnBudget,PlannerTrialSpec
from kaa.idol_road.preparation import prepare_battle,PreparationIssueCode
from kaa.idol_road.controller import IdolRoadController
from kaa.idol_road.models import Observation,Scene,Action
from kaa.idol_road.objectives import StageObjectives,read_rank_requirement
from kaa.idol_road.battle_observation import BattleObservationSession


def reads(text, confidence=.99):
    return tuple(RuleTextRead('frame',kind,(RuleTextLine(text,confidence),)) for kind in ('original','grayscale'))


def row():
    return VisibleStageRule((40,640,637,185),16,reads('16ターン目'),reads('unknown body',.8),
        turn_addition=RuleSymbolEvidence('frame','turn_addition',(385,749,225,43),.99))


def rules(current=None):
    return StageRules('ttmr',20,(RuleViewport('page',(current or row(),)),))


def formation():
    return Observation(Scene.CONFIRM,'ttmr',20,stars=0,formation_ready=True,total_turns=20,
                        formation_recommended_effect='concentration')


def goals():
    return StageObjectives('ttmr',20,tuple(read_rank_requirement(i,f'{4-i}位以上でクリア') for i in (1,2,3)))


def test_visible_marker_flags_incompatible_budget_without_predicting_reward():
    result=fixed_turn_budget_conflicts('ttmr',20,rules())
    assert len(result)==1 and result[0].declared_turn==16 and result[0].reason=='visible_turn_addition'
    assert not hasattr(result[0],'extra_turns')
    prepared=prepare_battle('ttmr',20,formation(),stage_rules=rules())
    assert prepared.turn_budget_conflicts==result
    assert PreparationIssueCode.TURN_BUDGET_UNSUPPORTED in {issue.code for issue in prepared.issues}
    assert not prepared.automatic_battle_ready


@pytest.mark.parametrize('change',[{'frame_id':'old'},{'score':.96},{'kind':'extra_play'},
    {'rect':(0,0,225,43)},{'rect':(385,749,0,43)}])
def test_stale_low_confidence_other_symbol_and_outside_card_are_not_confirmed(change):
    current=row()
    assert not fixed_turn_budget_conflicts('ttmr',20,rules(replace(current,
        turn_addition=replace(current.turn_addition,**change))))


@pytest.mark.parametrize('change',[{'stage':19},{'identity_source':'history'},{'source':'database'}])
def test_conflicts_require_current_stage_and_formation_source(change):
    assert not fixed_turn_budget_conflicts('ttmr',20,replace(rules(),**change))


def test_title_and_description_must_belong_to_the_same_frame_as_marker():
    current=replace(row(),turn_reads=reads('16ターン目',.97))
    assert not fixed_turn_budget_conflicts('ttmr',20,rules(current))
    current=replace(row(),description_reads=tuple(replace(read,frame_id='other') for read in reads('body')))
    assert not fixed_turn_budget_conflicts('ttmr',20,rules(current))


def test_trial_preflight_rejects_visible_extra_turns_before_challenge_request():
    with pytest.raises(UnsupportedTurnBudget,match='未发出挑战'):
        prepare_planner_trial(formation(),goals(),rules())
    flow=IdolRoadController(IdolRoadConfig(mode='planner_trial',characters=['ttmr']))
    flow._objectives[('ttmr',20)]=goals()
    flow._rules[('ttmr',20)]=rules()
    stopped=flow.step(formation())
    assert stopped.action==Action.STOP and stopped.reason=='planner_trial_dynamic_turns_unsupported'
    assert flow.battles_started==0 and not flow.awaiting_battle_start


def test_preview_and_manual_modes_remain_available():
    for mode,expected in (('preview',Action.READY),('manual',Action.START_BATTLE)):
        flow=IdolRoadController(IdolRoadConfig(mode=mode,characters=['ttmr']))
        flow._objectives[('ttmr',20)]=goals()
        flow._rules[('ttmr',20)]=rules()
        assert flow.step(formation()).action==expected


def test_direct_adapter_cannot_bypass_turn_budget_preflight():
    from kaa.idol_road.planner_adapter import run_planner_trial
    session=BattleObservationSession('battle','ttmr',20,20,'confirmed_formation_start_request')
    with pytest.raises(UnsupportedTurnBudget):
        run_planner_trial(PlannerTrialSpec('ttmr',20,3,20),None,session,
                          deadline=100,observe=lambda *_:None,stage_rules=rules())
