from dataclasses import replace
import pytest
from kaa.db.skill_card import SkillCard
from kaa.db.constants import ProduceExamEffectType
from kaa.idol_road.good_condition_priority import GoodConditionPriority
from kaa.idol_road.good_condition_rule import GoodConditionRequirement
from kaa.idol_road.field_priority import FieldPriority
from kaa.idol_road.planner_trial import prepare_planner_trial
from kaa.idol_road.formation import FormationSlot, FormationSlots
from kaa.idol_road.models import Observation, Scene
from kaa.idol_road.objectives import StageObjectives, read_rank_requirement
from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluation, EvaluationConfidence
from kaa.tasks.produce.new.play_cards.planner.state import BattleState, ExamBuffState


class Scores:
    def evaluate(self,card,state):
        return CardEvaluation(card._id,1000 if card.category == 'ProduceCardCategory_ActiveSkill' else 1,
                              EvaluationConfidence.HIGH,'test')


def state(value=4, **changes):
    return replace(BattleState(remaining_turns=19,hp=50,genki=0,is_exam=True,
        buffs=ExamBuffState(parameter_buff_turns=value),observed_effect_values=(('parameter_buff',value),)),**changes)


def cards():
    active = SkillCard.from_id('p_card-01-act-1_001')
    mental = SkillCard.from_id('p_card-01-men-0_007')
    assert active is not None and mental is not None
    return active,mental


def policy(*goals):
    return GoodConditionPriority(20,goals or (GoodConditionRequirement(6,4,('frame',)),),evaluator=Scores())


def test_current_redraw_can_build_duration_without_certifying_visible_followup():
    from kaa.idol_road.category_priority import supports_visible_followup
    redraw=SkillCard.from_id('p_card-01-sup-3_109')
    assert redraw is not None
    redraw=SkillCard.from_asset_id(redraw._asset_id,1)
    assert redraw is not None
    selected=policy();selected.allow_current_effects=True
    result=selected.select(state(),(redraw,))
    assert result.preferred_indices==(0,) and 'estimated_progress=3' in result.reason
    assert not supports_visible_followup(redraw,state())


def test_current_four_turns_is_not_four_turns_at_stage_six():
    before = state()
    result = policy().select(before,cards())
    assert result.preferred_indices == (1,)
    assert 'current_reserve=8' in result.reason and 'declared_turn=6' in result.reason
    assert 'phase_unconfirmed' in result.reason and 'reward_unconfirmed' in result.reason
    assert before.buffs.parameter_buff_turns == 4 and before.observed_effect_values == (('parameter_buff',4),)


def test_reserve_accounts_for_each_remaining_transition():
    result = policy().select(state(4,remaining_turns=16),cards())
    assert result.preferred_indices == (1,) and 'current_reserve=5' in result.reason
    prepared = policy().select(state(8),cards())
    assert not prepared.preferred_indices
    past = policy().select(state(4,remaining_turns=15),cards())
    assert not past.preferred_indices


def test_known_payment_can_be_avoided_using_a_visible_nonspending_alternative():
    active,_ = cards()
    spender = replace(active,cost_type='ExamCostType_ExamParameterBuff',cost_value=2)
    current = state(8)
    result = policy().select(current,(spender,active))
    assert result.preferred_indices == (1,) and 'protect_duration=8' in result.reason
    assert not result.end_turn
    assert not policy().select(current,(spender,)).end_turn
    assert not policy().select(current,(spender,)).preferred_indices


@pytest.mark.parametrize('changes', [
    {'remaining_turns_known':False}, {'observed_effect_values':()},
    {'observed_effect_values':(('parameter_buff',None),)},
    {'observed_effect_values':(('parameter_buff',5),)},
])
def test_unknown_or_conflicting_duration_does_not_become_budget(changes):
    assert not policy().select(state(**changes),cards()).preferred_indices


def test_current_only_mode_keeps_unknown_duration_advisory_without_claiming_field_condition():
    selected=policy();selected.allow_unknown_duration=True
    current=state(observed_effect_values=(('parameter_buff',None),))
    result=selected.select(current,cards())
    assert not result.stop and not result.preferred_indices and result.allowed_indices is None
    assert 'condition_unconfirmed' in result.reason
    assert current.observed_effect_values==(('parameter_buff',None),)
    assert selected.select(replace(current,remaining_turns_known=False),cards()).stop


def test_prepared_early_goal_does_not_hide_later_unmet_duration():
    selected = policy(GoodConditionRequirement(3,1,('early',)),GoodConditionRequirement(6,4,('later',)))
    result = selected.select(state(4),cards())
    assert result.preferred_indices == (1,) and 'declared_turn=6' in result.reason


def test_conflicting_thresholds_cannot_be_turned_into_favorable_requirement():
    selected = policy(GoodConditionRequirement(6,4,('a',)),GoodConditionRequirement(6,5,('b',)))
    assert not selected.select(state(),cards()).preferred_indices


def test_field_coordinator_uses_duration_policy_without_requiring_concentration():
    selected = FieldPriority(20,(),(),(GoodConditionRequirement(6,4,('frame',)),))
    selected.good_condition.evaluator = Scores()
    assert selected.select(state(),cards()).preferred_indices == (1,)


def test_current_available_first_card_does_not_claim_known_future_costs():
    before = state(observed_effects=frozenset({'parameter_buff','stamina_consumption_modifier'}),
                   observed_effect_values=(('parameter_buff',4),('stamina_consumption_modifier',None)))
    assert not policy().select(before,cards()).preferred_indices
    assert policy().select(before,cards(),current_cards_available=True).preferred_indices == (1,)


def test_current_recommendation_selects_good_profile_instead_of_concentration():
    slots = FormationSlots('frame',(FormationSlot('main',(1,1,100,50),'occupied',100),
                                    FormationSlot('sub',(101,1,100,50),'occupied',100)))
    formation = Observation(Scene.CONFIRM,'ttmr',19,stars=0,formation_ready=True,total_turns=20,
                            formation_slots=slots,formation_recommended_effect='good_condition')
    goals = StageObjectives('ttmr',19,tuple(read_rank_requirement(i,f'{4-i}位以上でクリア') for i in (1,2,3)))
    spec = prepare_planner_trial(formation,goals)
    assert spec.archetype == ProduceExamEffectType.ExamParameterBuff
    assert spec.scope == 'single_good_condition_trial'
    concentration = prepare_planner_trial(replace(formation,formation_recommended_effect='concentration'),goals)
    assert concentration.archetype == ProduceExamEffectType.ExamLessonBuff
    assert concentration.scope == 'single_concentration_trial'
    for value in (None,'review'):
        with pytest.raises(ValueError):
            prepare_planner_trial(replace(formation,formation_recommended_effect=value),goals)
    with pytest.raises(ValueError,match='槽位'):
        prepare_planner_trial(replace(formation,formation_slots=None),goals)


def test_shared_simulation_extends_existing_duration_instead_of_taking_maximum():
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play
    _,mental = cards()
    before = state(4)
    after = simulate_play(before,mental)
    assert after.buffs.parameter_buff_turns == 6
    assert dict(after.effect_values_for_planning)['parameter_buff'] == 6
    assert before.buffs.parameter_buff_turns == 4
