from dataclasses import replace
import pytest
from kaa.db.skill_card import SkillCard
from kaa.idol_road.native_payment import supports_native_payment_conflict
from kaa.idol_road.hp_safety import HpSafetyPolicy
from kaa.idol_road.hp_preparation import HpPreparationBudget
from kaa.idol_road.category_priority import supports_visible_followup
from kaa.tasks.produce.new.play_cards.planner.simulator import can_afford_play,simulate_play
from kaa.tasks.produce.new.play_cards.planner.evaluator import CompositeEvaluator
from tests.kaa.test_idol_road_hp_safety import state


def test_native_fee_conflict_does_not_guess_an_upgrade_or_numeric_payment():
    card=SkillCard.from_id('p_card-01-ido-3_014')
    before=replace(state(turn=9,hp=0),genki=5)
    assert can_afford_play(before,card) is False and supports_native_payment_conflict(card)
    current=replace(before,native_payment_conflict_card_ids=frozenset({card._id}))
    assert can_afford_play(current,card) is None
    guard=HpSafetyPolicy(20,HpPreparationBudget((),None,False),reserve_mode='prefer')
    assert guard.select(current,(card,)).allowed_indices==(0,)
    review=guard.review.candidates[0]
    assert review.affordable is None and review.projected_hp is None and not review.projected_hp_known
    assert 'payment_model_conflict' in review.reason
    assert not HpSafetyPolicy(20,HpPreparationBudget((),None,False)).select(current,(card,)).allowed_indices
    after=simulate_play(current,card)
    assert not after.hp_known and not after.genki_known and after.playable_count==current.playable_count-1
    assert not supports_visible_followup(card,current)
    assert CompositeEvaluator().evaluate(card,current).confirmed_output==0
    assert card.upgrade_count==0 and card.stamina==7


@pytest.mark.parametrize('key',['p_card-01-act-2_001','p_card-01-sup-3_109','p_card-01-men-2_034'])
def test_draw_extra_actions_and_installed_triggers_do_not_use_fee_conflict_contract(key):
    assert not supports_native_payment_conflict(SkillCard.from_id(key))


def test_modified_or_forced_hp_definition_cannot_supply_action_invariant_audit():
    card=SkillCard.from_id('p_card-01-ido-3_014')
    assert not supports_native_payment_conflict(replace(card,force_stamina=1))
    assert not supports_native_payment_conflict(replace(card,cost_value=1))
