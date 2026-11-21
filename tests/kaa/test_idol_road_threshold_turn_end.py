from dataclasses import replace
import pytest
from kaa.idol_road.threshold_turn_end import ThresholdTurnEndEvidence,interpret_threshold_turn_end
from kaa.idol_road.rules import RuleTextRead,RuleTextLine,VisibleStageRule,RuleViewport,StageRules
from kaa.idol_road.concentration_priority import concentration_objectives
from kaa.idol_road.rule_catalog import build_rule_catalog
from kaa.idol_road.rule_conditions import RuleConditionObservation,evaluate_rule_conditions,ConditionVerdict

def reads(text,frame='frame',confidence=.99):
    return tuple(RuleTextRead(frame,kind,(RuleTextLine(text,confidence),)) for kind in ('original','grayscale'))

def evidence(kind='concentration_two'):
    return ThresholdTurnEndEvidence('frame',kind,(0,0,100,80),.99,.99,(20,1,30,25),(30,40,30,25),
                                   reads('3' if kind=='concentration_two' else '19'),reads('+2' if kind=='concentration_two' else '+5'))

@pytest.mark.parametrize('kind,turn,threshold,reward,amount',[
    ('concentration_two',3,3,'concentration',2),('score_five',6,19,'score',5)])
def test_supported_patterns_keep_condition_reward_and_execution_separate(kind,turn,threshold,reward,amount):
    checked=interpret_threshold_turn_end(turn,evidence(kind),reads('raw low-confidence text',confidence=.9))
    declaration=checked.declaration
    assert declaration is not None
    assert (declaration.start_turn,declaration.concentration_at_least,declaration.reward_kind,declaration.amount)==(turn,threshold,reward,amount)
    assert not declaration.executable
    row=VisibleStageRule((1,1,100,100),turn,reads(f'{turn}ターン目'),reads('raw low-confidence text',confidence=.9),threshold_turn_end=checked)
    rules=StageRules('ttmr',19,(RuleViewport('viewport',(row,row)),))
    goals=concentration_objectives('ttmr',19,rules)
    assert len(goals)==1 and goals[0].deadline_turn==turn-1
    catalog=build_rule_catalog('ttmr',19,rules)
    observation=RuleConditionObservation('ttmr',19,'battle','current',turn,concentration=threshold,source='confirmed_current_battle_observation')
    report=evaluate_rule_conditions(catalog,observation,battle_id='battle')
    assert all(check.verdict==ConditionVerdict.SATISFIED for check in report.checks)
    assert not report.executable and not report.trigger_phase_confirmed

@pytest.mark.parametrize('change',[
    {'template_score':.969},{'comparator_score':.969},{'threshold_reads':reads('4')},
    {'gain_reads':reads('-2')},{'gain_reads':reads('+3')},{'threshold_reads':reads('3',confidence=.979)},
    {'threshold_reads':reads('3',frame='old')},{'gain_rect':(30,10,30,25)},
    {'threshold_rect':(-1,1,30,25)},{'threshold_reads':(reads('3')[0],reads('9')[1])}])
def test_partial_pattern_unknown_digits_sign_or_geometry_never_become_declarations(change):
    assert interpret_threshold_turn_end(3,replace(evidence(),**change),reads('raw',confidence=.9)).declaration is None

@pytest.mark.parametrize('text',['集中が4以上の場合、集中+2','集中が3以下の場合、集中+2',
    '集中が3以上の場合、集中-2','集中が3以上の場合、スコア+2',
    '集中が3以上の場合、以降、ターン開始時、集中+2'])
def test_high_confidence_original_conflict_is_not_replaced_by_template(text):
    checked=interpret_threshold_turn_end(3,evidence(),reads(text))
    assert checked.status=='high_confidence_conflict' and checked.declaration is None
