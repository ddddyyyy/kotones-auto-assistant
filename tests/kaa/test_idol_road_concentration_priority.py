from dataclasses import replace
import pytest
from kaa.db.skill_card import SkillCard
from kaa.idol_road.rules import RuleTextRead, RuleTextLine, RuleViewport, StageRules, VisibleStageRule
from kaa.idol_road.rule_symbols import RuleSymbolEvidence
from kaa.idol_road.concentration_rule import ConcentrationRuleEvidence, interpret_concentration_rule, ConcentrationDoubleDeclaration
from kaa.idol_road.concentration_priority import concentration_objectives, ConcentrationObjective, ConcentrationPriority
from kaa.idol_road.rule_catalog import build_rule_catalog
from kaa.idol_road.rule_conditions import RuleConditionObservation, evaluate_rule_conditions, ConditionVerdict
from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluation, EvaluationConfidence
from kaa.tasks.produce.new.play_cards.planner.state import BattleState, ExamBuffState

TEXT = '集中が42以上の場合、集中2倍'

def reads(text=TEXT, frame='frame', confidence=.99):
    return tuple(RuleTextRead(frame, kind,(RuleTextLine(text,confidence),)) for kind in ('original','grayscale'))

def evidence(text=TEXT):
    return ConcentrationRuleEvidence('frame',(
        RuleSymbolEvidence('frame','concentration',(0,0,20,20),.99),
        RuleSymbolEvidence('frame','concentration',(0,40,20,20),.99)),reads(text))

def rule():
    analysis=interpret_concentration_rule(10,evidence(),reads())
    return VisibleStageRule((1,1,100,100),10,reads('10ターン目'),reads(),concentration_rule=analysis)

def stage_rules(*rows):
    return StageRules('ttmr',19,(RuleViewport('viewport',rows or (rule(),)),))

def state(value=10, **changes):
    return replace(BattleState(remaining_turns=15,hp=50,genki=0,is_exam=True,
                              buffs=ExamBuffState(lesson_buff=value),observed_effect_values=(('lesson_buff',value),)),**changes)

class Scores:
    def evaluate(self,card,state):
        score=1000 if card.category=='ProduceCardCategory_ActiveSkill' else 1
        return CardEvaluation(card._id,score,EvaluationConfidence.HIGH,'test')

def policy():
    return ConcentrationPriority(20,(ConcentrationObjective(ConcentrationDoubleDeclaration(10,42),('frame',)),),evaluator=Scores())

def test_rule_is_a_declaration_without_trigger_phase_or_effect_execution():
    analysis=interpret_concentration_rule(10,evidence(),reads())
    assert analysis.declaration==ConcentrationDoubleDeclaration(10,42)
    assert not analysis.declaration.executable
    goals=concentration_objectives('ttmr',19,stage_rules())
    assert len(goals)==1 and goals[0].deadline_turn==9 and 'phase_unconfirmed' in goals[0].timing
    assert not concentration_objectives('ttmr',18,stage_rules())
    assert not concentration_objectives('ttmr',19,replace(stage_rules(),identity_source='history'))

@pytest.mark.parametrize('change',[
    {'frame_id':'old'}, {'reads':reads(confidence=.979)}, {'icons':evidence().icons[:1]},
    {'icons':(evidence().icons[0],replace(evidence().icons[1],score=.969))},
    {'icons':(evidence().icons[0],evidence().icons[0])},
    {'reads':(reads()[0],reads(TEXT.replace('42','43'))[1])}])
def test_missing_low_confidence_or_conflicting_evidence_does_not_become_goal(change):
    assert interpret_concentration_rule(10,replace(evidence(),**change),reads()).declaration is None

@pytest.mark.parametrize('text',[TEXT.replace('以上','以下'),TEXT.replace('2倍','3倍'),TEXT+'さらに抽出'])
def test_unsupported_full_text_is_not_simplified(text):
    assert interpret_concentration_rule(10,evidence(text),reads(text)).declaration is None

def test_original_high_confidence_conflict_overrides_favorable_masked_text():
    bad=interpret_concentration_rule(10,evidence(),reads(TEXT.replace('42','43')))
    assert bad.status=='high_confidence_conflict'
    conflicting=replace(rule(),concentration_rule=bad)
    assert not concentration_objectives('ttmr',19,stage_rules(rule(),conflicting))
    goals=concentration_objectives('ttmr',19,stage_rules(rule(),rule()))
    assert len(goals)==1 and goals[0].evidence_frames==('frame',)


@pytest.mark.parametrize('value,verdict',[(41,ConditionVerdict.NOT_SATISFIED),(42,ConditionVerdict.SATISFIED),
    (84,ConditionVerdict.SATISFIED),(None,ConditionVerdict.UNKNOWN)])
def test_actual_numeric_condition_remains_separate_from_observed_reward(value,verdict):
    catalog=build_rule_catalog('ttmr',19,stage_rules())
    observation=RuleConditionObservation('ttmr',19,'battle','current',10,concentration=value,
                                         source='confirmed_current_battle_observation')
    result=evaluate_rule_conditions(catalog,observation,battle_id='battle')
    assert result.checks[0].verdict==verdict
    assert not result.trigger_phase_confirmed and not result.executable
    early=evaluate_rule_conditions(catalog,replace(observation,turn=9),battle_id='battle')
    assert early.checks[0].verdict==ConditionVerdict.UNKNOWN


def test_preparation_can_choose_lower_output_for_more_verified_concentration():
    active=SkillCard.from_id('p_card-01-act-1_001')
    mental=SkillCard.from_id('p_card-01-men-2_035')
    assert active is not None and mental is not None
    before=state()
    result=policy().select(before,(active,mental))
    assert result.preferred_indices==(1,)
    assert 'prepare_by=9' in result.reason and 'reward_unconfirmed' in result.reason
    assert before.buffs.lesson_buff==10
    assert not policy().select(state(42),(active,mental)).preferred_indices
    assert not policy().select(state(remaining_turns=11),(active,mental)).preferred_indices
    assert not policy().select(state(observed_effect_values=(('lesson_buff',None),)),(active,mental)).preferred_indices
    assert not policy().select(state(remaining_turns_known=False),(active,mental)).preferred_indices


def test_timing_and_hud_values_cannot_be_replaced_by_historical_or_inferred_values():
    active=SkillCard.from_id('p_card-01-ido-3_014')
    assert active is not None
    assert not policy().select(state(observed_effect_values=(('lesson_buff',11),)),(active,)).preferred_indices
    assert not ConcentrationPriority(20,()).select(state(),(active,)).preferred_indices


def test_visible_first_play_remains_legal_with_unknown_hp_modifier_but_does_not_prove_future_costs():
    from kaa.tasks.produce.new.play_cards.planner.simulator import can_afford_play,simulate_play
    mental=SkillCard.from_id('p_card-01-men-0_008')
    active=SkillCard.from_id('p_card-01-act-2_001')
    assert mental is not None and active is not None
    current=state(34,observed_effects=frozenset({'lesson_buff','stamina_consumption_modifier'}),
                  observed_effect_values=(('lesson_buff',34),('stamina_consumption_modifier',None)))
    assert can_afford_play(current,mental) is None
    assert not policy().select(current,(mental,)).preferred_indices
    result=policy().select(current,(active,mental),current_cards_available=True)
    assert result.preferred_indices==(1,) and 'current_hand_available=True' in result.reason
    future=simulate_play(current,mental)
    assert not future.hp_known
    assert can_afford_play(future,mental) is None
    assert dict(current.observed_effect_values)['stamina_consumption_modifier'] is None


def test_observed_first_play_cannot_override_known_unaffordability_or_unknown_special_payment():
    from kaa.idol_road.category_priority import supports_first_play
    active=SkillCard.from_id('p_card-01-act-2_001')
    assert active is not None
    assert not supports_first_play(state(2),active,observed_available=True)
    assert not supports_first_play(state(observed_effect_values=()),active,observed_available=True)


def test_reached_early_target_does_not_hide_next_unmet_threshold():
    from kaa.idol_road.threshold_turn_end import ThresholdTurnEndDeclaration
    goals=(ConcentrationObjective(ThresholdTurnEndDeclaration(3,3,'concentration',2),('first',)),
           ConcentrationObjective(ThresholdTurnEndDeclaration(6,19,'score',5),('second',)),
           ConcentrationObjective(ConcentrationDoubleDeclaration(10,42),('third',)))
    selected=ConcentrationPriority(20,goals,evaluator=Scores())
    active=SkillCard.from_id('p_card-01-act-1_001')
    mental=SkillCard.from_id('p_card-01-men-2_035')
    assert active is not None and mental is not None
    result=selected.select(state(10,remaining_turns=19),(active,mental))
    assert result.preferred_indices==(1,) and 'threshold=19,prepare_by=5' in result.reason


@pytest.mark.parametrize('value',[40,42])
def test_preserve_stock_when_only_alternative_spends_concentration(value):
    spender=SkillCard.from_id('p_card-01-act-2_001')
    neutral=SkillCard.from_id('p_card-01-act-1_001')
    assert spender is not None and neutral is not None
    result=policy().select(state(value),(spender,neutral))
    assert result.preferred_indices==(1,) and f'protect_concentration={value}' in result.reason
    assert not result.end_turn
    only_spender=policy().select(state(value),(spender,))
    assert not only_spender.preferred_indices and not only_spender.end_turn
