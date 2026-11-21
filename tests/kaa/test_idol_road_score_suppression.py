from dataclasses import replace
from types import SimpleNamespace
from kaa.db.skill_card import SkillCard
from kaa.idol_road.score_suppression import CurrentScoreSuppression,SuppressionAwareEvaluator,score_suppression_active,score_only_card
from kaa.idol_road.ranked_field_policy import RankedFieldPolicy
from kaa.tasks.produce.new.play_cards.planner.selection_policy import SelectionConstraint
from kaa.tasks.produce.new.play_cards.planner.state import ExamBuffState
from tests.kaa.test_idol_road_hp_safety import state


def current():
    return replace(state(turn=2,hp=50),buffs=ExamBuffState(lesson_buff=10,parameter_buff_turns=5),
        observed_effect_values=(('lesson_buff',10),('parameter_buff',5)))


def cue():return CurrentScoreSuppression('frame-2',(28,308,31,28),.999,.5,True)
def scorer():return SkillCard.from_id('p_card-01-act-1_020')
def field(choice=None):return SimpleNamespace(select=lambda *args:choice or SelectionConstraint())


def test_current_score_block_zeroes_direct_output_without_mutating_original_utility_card():
    card=SkillCard.from_id('p_card-01-ido-3_014');effects=card.play_effects
    evaluator=SuppressionAwareEvaluator();normal=evaluator.evaluate(card,current())
    assert normal.projected_output>0
    evaluator.current=cue();blocked=evaluator.evaluate(card,current())
    assert blocked.projected_output==blocked.confirmed_output==0 and 'スランプ' in blocked.reason
    assert card.play_effects is effects and len(card.play_effects)==2
    assert not score_only_card(card) and score_only_card(scorer())


def test_single_use_scorer_is_preserved_but_a_visible_setup_still_plays():
    evaluator=SuppressionAwareEvaluator();evaluator.current=cue()
    policy=RankedFieldPolicy(field(),evaluator=evaluator)
    result=policy.select(current(),(scorer(),SkillCard.from_id('p_card-01-men-2_035')))
    assert result.allowed_indices==result.preferred_indices==(1,)
    assert policy.trace.score_suppression==cue()
    only=policy.select(current(),(scorer(),))
    assert only.end_turn and not only.stop and only.allowed_indices==()
    assert 'preserve_pure_scoring' in only.reason


def test_explicit_field_precondition_can_spend_the_single_use_scorer():
    evaluator=SuppressionAwareEvaluator();evaluator.current=cue()
    policy=RankedFieldPolicy(field(SelectionConstraint((0,),reason='field_goal:current_last_A')),evaluator=evaluator)
    result=policy.select(current(),(scorer(),))
    assert result.preferred_indices==(0,) and not result.end_turn
    assert policy.trace.evaluations[0][1].projected_output==0


def test_current_block_does_not_discard_a_play_for_sleep_or_non_single_use_setup():
    evaluator=SuppressionAwareEvaluator();evaluator.current=cue()
    policy=RankedFieldPolicy(field(),evaluator=evaluator)
    for key in ('p_card-00-acc-0_002','p_card-01-men-0_007'):
        result=policy.select(current(),(SkillCard.from_id(key),))
        assert result.allowed_indices==result.preferred_indices==(0,)
        assert not result.stop and not result.end_turn


def test_stale_missing_or_wrong_column_evidence_does_not_prove_an_active_block():
    evaluator=SuppressionAwareEvaluator();policy=RankedFieldPolicy(field(),evaluator=evaluator)
    for value in (None,replace(cue(),frame_id='old'),replace(cue(),rect=(28,146,31,28)),replace(cue(),red_fraction=0)):
        evaluator.current=value
        assert not score_suppression_active(value,current())
        result=policy.select(current(),(scorer(),))
        assert result.preferred_indices==(0,) and not result.end_turn
        assert policy.trace.score_suppression is None


def test_recyclable_pure_scorer_is_also_preserved_while_current_score_is_blocked():
    basic=SkillCard.from_id('p_card-00-act-0_001')
    assert basic.play_move_position_type=='ProduceCardMovePositionType_Grave' and score_only_card(basic)
    evaluator=SuppressionAwareEvaluator();evaluator.current=cue()
    policy=RankedFieldPolicy(field(),evaluator=evaluator)
    result=policy.select(current(),(basic,))
    assert result.end_turn and not result.stop and result.allowed_indices==()
    evaluator.current=None
    ordinary=policy.select(current(),(basic,))
    assert not ordinary.end_turn and ordinary.preferred_indices==(0,)
