from dataclasses import replace
import pytest
from kaa.db.skill_card import SkillCard
from kaa.db.constants import ProducePlanType,ProduceExamEffectType
from kaa.idol_road.ranked_field_policy import CurrentCardPreview,RankedFieldPolicy
from kaa.tasks.produce.new.play_cards.planner.state import BattleState,ExamBuffState
from kaa.tasks.produce.new.play_cards.planner.evaluator import CompositeEvaluator
from kaa.tasks.produce.new.play_cards.planner.visible_output import current_preview_output
from kaa.battle_turn import AbsoluteTurnObservation
from tests.kaa.test_idol_road_ranked_field_policy import delegate


def scoring_card():
    base=SkillCard.from_id('p_card-01-act-1_020')
    assert base is not None
    return base


def current():
    return BattleState(5,8,2,is_exam=True,plan_type=ProducePlanType.Plan1,
        archetype=ProduceExamEffectType.ExamParameterBuff,buffs=ExamBuffState(lesson_buff=35,parameter_buff_turns=29),
        observed_effect_values=(('lesson_buff',35),('parameter_buff',29)),
        absolute_turn_observation=AbsoluteTurnObservation('battle','frame',14,5))


def test_single_hit_preview_ranks_without_fabricating_attribute_multiplier_or_total_gain():
    card=scoring_card();before=current();policy=RankedFieldPolicy(delegate())
    policy.current_previews=(CurrentCardPreview('frame',card._id,card.upgrade_count,5319,None),)
    result=policy.select(before,(card,))
    evaluation=policy.trace.evaluations[0][1]
    assert result.preferred_indices==(0,)
    assert evaluation.projected_output==5319 and evaluation.confirmed_output==0
    assert '实际总收益未确认' in evaluation.reason
    assert before.exam_score_percent is None and before.observed_card_output is None


@pytest.mark.parametrize('changes',[
    {'frame_id':'old'},{'card_id':'other'},{'upgrade':99},{'per_hit':-1},{'per_hit':1000000},
    {'per_hit':True},{'count':2},{'count':True},
])
def test_stale_wrong_version_invalid_or_multiple_count_evidence_is_not_used(changes):
    card=scoring_card();before=current();policy=RankedFieldPolicy(delegate())
    policy.current_previews=(replace(CurrentCardPreview('frame',card._id,card.upgrade_count,5319,None),**changes),)
    policy.select(before,(card,))
    assert policy.trace.evaluations[0][1]==CompositeEvaluator().evaluate(card,before)


def test_duplicate_card_or_preview_identity_is_not_mapped_by_id_alone():
    card=scoring_card();policy=RankedFieldPolicy(delegate())
    preview=CurrentCardPreview('frame',card._id,card.upgrade_count,5319,None)
    policy.current_previews=(preview,preview)
    policy.select(current(),(card,))
    assert policy.trace.evaluations[0][1].projected_output!=5319
    policy.current_previews=(preview,)
    policy.select(current(),(card,card))
    assert all(e.projected_output!=5319 for _,e in policy.trace.evaluations)


@pytest.mark.parametrize('change',[
    {'is_exam':False},{'plan_type':ProducePlanType.Plan2},
    {'projected_effect_values':(('lesson_buff',35),)},
    {'unresolved_growth_card_ids':frozenset({'p_card-01-act-1_020'})},
])
def test_lower_bound_scope_does_not_open_other_modes_or_future_states(change):
    card=scoring_card();state=replace(current(),observed_card_output=(card._id,20079,None),
        observed_card_output_is_lower_bound=True,**change)
    assert current_preview_output(card,state) is None


def test_wide_single_hit_value_does_not_change_legacy_preview_limit():
    card=scoring_card();state=replace(current(),observed_card_output=(card._id,20079,None))
    assert current_preview_output(card,state) is None
    assert current_preview_output(card,replace(state,observed_card_output_is_lower_bound=True))==20079
