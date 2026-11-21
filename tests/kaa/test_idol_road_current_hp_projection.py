from dataclasses import replace
import pytest
from kaa.db.skill_card import SkillCard
from kaa.db.constants import ProduceExamEffectType
from kaa.idol_road.current_hp_projection import supports_current_hp_projection
from kaa.idol_road.category_priority import supports_visible_followup
from kaa.idol_road.hp_safety import HpSafetyPolicy
from kaa.tasks.produce.new.play_cards.planner.state import ExamBuffState
from tests.kaa.test_idol_road_hp_safety import state, budget


def observed():
    return replace(state(turn=2, hp=50), genki=8, buffs=ExamBuffState(parameter_buff_turns=8),
        observed_effects=frozenset({'parameter_buff'}), observed_effect_values=(('parameter_buff', 8),))


def score_card():
    value = SkillCard.from_id('p_card-01-act-0_022')
    assert value is not None
    return value


def test_one_plain_score_only_play_can_be_hp_safe_without_becoming_a_two_card_prefix():
    current, card = observed(), score_card()
    assert not supports_visible_followup(card, current)
    assert supports_current_hp_projection(card, current)
    assert HpSafetyPolicy(20, budget()).select(current, (card,)).allowed_indices == (0,)


@pytest.mark.parametrize('change', [
    {'effect_type': ProduceExamEffectType.ExamCardDraw}, {'_chain_produce_exam_effect_id': 'unknown'},
    {'_produce_exam_status_enchant_id': 'unknown'}, {'_produce_card_status_enchant_id': 'unknown'},
    {'_produce_card_grow_effect_ids': '["unknown"]'}, {'_produce_card_search_id': 'unknown'},
    {'effect_turn': 1}, {'effect_count': 2},
])
def test_unknown_side_effects_and_other_primitives_stay_outside_the_new_support(change):
    card = score_card()
    effect = replace(card.play_effects[0].produce_exam_effect, **change)
    altered = replace(card, play_effects=[replace(card.play_effects[0], produce_exam_effect=effect)])
    assert not supports_current_hp_projection(altered, observed())


def test_unknown_trigger_growth_and_simulated_future_cannot_supply_current_support():
    card, current = score_card(), observed()
    for altered in (replace(current, unresolved_growth_card_ids=(card._id,)),
                    replace(current, projected_effect_values=())):
        assert not supports_current_hp_projection(card, altered)
    unknown = replace(card, _play_produce_exam_trigger_id='unknown')
    assert HpSafetyPolicy(20, budget()).select(current, (unknown,)).stop


def test_hp_payment_and_actual_condition_are_still_checked():
    card = score_card()
    assert HpSafetyPolicy(20, budget()).select(observed(), (replace(card, force_stamina=1),)).stop
    unknown = replace(observed(), observed_effect_values=(), observed_effects=frozenset(), buffs=ExamBuffState())
    assert HpSafetyPolicy(20, budget()).select(unknown, (card,)).stop


@pytest.mark.parametrize('card_id',['p_card-01-act-1_020','p_card-01-ido-2_017'])
def test_pure_concentration_or_conditional_score_branches_have_only_current_hp_support(card_id):
    current=observed();card=SkillCard.from_id(card_id)
    assert supports_current_hp_projection(card,current)
    guard=HpSafetyPolicy(20,budget())
    assert guard.select(current,(card,)).allowed_indices==(0,)
    assert guard.review.candidates[0].projected_hp==current.hp


def test_unknown_pure_score_trigger_stays_unknown_without_invalidating_known_payment():
    from kaa.tasks.produce.new.play_cards.planner.trigger import evaluate_trigger_id,TriggerVerdict
    card=SkillCard.from_id('p_card-01-act-1_020')
    segment=replace(card.play_effects[0],_produce_exam_trigger_id='unrecognized-score-condition')
    card=replace(card,play_effects=[segment])
    assert evaluate_trigger_id(segment._produce_exam_trigger_id,observed()) is TriggerVerdict.UNKNOWN
    assert supports_current_hp_projection(card,observed())
    assert HpSafetyPolicy(20,budget()).select(observed(),(card,)).allowed_indices==(0,)
    assert not supports_visible_followup(card,observed())


def test_a_resource_effect_in_an_optional_score_branch_is_still_rejected():
    card=SkillCard.from_id('p_card-01-ido-2_017')
    optional=card.play_effects[1]
    changed=replace(optional,produce_exam_effect=replace(optional.produce_exam_effect,
        effect_type=ProduceExamEffectType.ExamStaminaRecoverFix))
    assert not supports_current_hp_projection(replace(card,play_effects=[card.play_effects[0],changed]),observed())


@pytest.mark.parametrize('card_id,expected_hp,expected_genki',[
    ('p_card-01-act-1_036',50,4),('p_card-00-men-3_011',49,16)])
def test_modeled_great_condition_and_cost_reduction_have_current_payment_support(card_id,expected_hp,expected_genki):
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play
    current=observed();card=SkillCard.from_id(card_id)
    assert supports_current_hp_projection(card,current)
    assert not supports_visible_followup(card,current)
    after=simulate_play(current,card)
    assert after.hp_known and after.genki_known and (after.hp,after.genki)==(expected_hp,expected_genki)
    assert after.remaining_turns==current.remaining_turns
    assert HpSafetyPolicy(20,budget(),reserve_mode='prefer').select(current,(card,)).allowed_indices==(0,)


@pytest.mark.parametrize('card_id',[
    'p_card-01-men-2_034','p_card-01-men-2_038','p_card-01-ido-3_128',
    'p_card-01-act-3_049','p_card-00-sup-3_093','p_card-00-men-2_012'])
@pytest.mark.parametrize('upgrade',[0,1])
def test_current_draw_and_resource_neutral_installations_do_not_require_future_trigger_simulation(card_id,upgrade):
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play
    base=SkillCard.from_id(card_id)
    card=SkillCard.from_asset_id(base._asset_id,upgrade)
    current=replace(observed(),buffs=ExamBuffState(lesson_buff=20,parameter_buff_turns=8),
        observed_effects=frozenset({'lesson_buff','parameter_buff'}),
        observed_effect_values=(('lesson_buff',20),('parameter_buff',8)))
    assert supports_current_hp_projection(card,current)
    assert not supports_visible_followup(card,current)
    after=simulate_play(current,card)
    assert after.hp_known and after.hp>=0 and after.remaining_turns==current.remaining_turns
    guard=HpSafetyPolicy(20,budget(),reserve_mode='prefer')
    assert guard.select(current,(card,)).allowed_indices==(0,)


@pytest.mark.parametrize('change',[
    {'effect_type':ProduceExamEffectType.ExamStaminaRecoverFix},
    {'_chain_produce_exam_effect_id':'unresolved'}, {'_produce_exam_status_enchant_id':'unresolved'},
    {'effect_value1':-1}, {'_produce_card_grow_effect_ids':'["unresolved"]'},
])
def test_installed_effects_with_unreviewed_resource_or_nested_changes_are_rejected(monkeypatch,change):
    from kaa.db.exam_status_enchant import get_exam_status_enchant
    import kaa.idol_road.current_hp_projection as projection
    card=SkillCard.from_id('p_card-01-men-2_038')
    installed=get_exam_status_enchant(card.play_effects[-1].produce_exam_effect._produce_exam_status_enchant_id)
    altered=replace(installed,effects=(replace(installed.effects[0],**change),))
    monkeypatch.setattr(projection,'get_exam_status_enchant',lambda _:altered)
    assert not supports_current_hp_projection(card,observed())


def test_missing_installation_roots_or_delayed_effects_are_not_silently_ignored(monkeypatch):
    import kaa.idol_road.current_hp_projection as projection
    from kaa.db.exam_status_enchant import get_exam_status_enchant
    card=SkillCard.from_id('p_card-01-men-2_038')
    installed=get_exam_status_enchant(card.play_effects[-1].produce_exam_effect._produce_exam_status_enchant_id)
    monkeypatch.setattr(projection,'get_exam_status_enchant',lambda _:replace(installed,root_effects_complete=False))
    assert not supports_current_hp_projection(card,observed())
    base=SkillCard.from_id('p_card-01-act-3_049');card=SkillCard.from_asset_id(base._asset_id,1)
    monkeypatch.setattr(projection,'load_exam_effects',lambda _: {})
    assert not supports_current_hp_projection(card,observed())


@pytest.mark.parametrize('upgrade',[0,1])
def test_installing_one_next_card_repetition_does_not_predict_the_following_card(upgrade):
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play
    base=SkillCard.from_id('p_card-01-men-3_006');card=SkillCard.from_asset_id(base._asset_id,upgrade)
    current=observed()
    assert supports_current_hp_projection(card,current)
    assert not supports_visible_followup(card,current)
    after=simulate_play(current,card)
    assert after.playable_count==current.playable_count and after.remaining_turns==current.remaining_turns
    effect=card.play_effects[-2].produce_exam_effect
    for change in ({'effect_value1':2},{'effect_count':2},{'_produce_card_search_id':'unknown'},
                   {'_chain_produce_exam_effect_id':'unknown'}):
        altered=replace(card,play_effects=[*card.play_effects[:-2],
            replace(card.play_effects[-2],produce_exam_effect=replace(effect,**change)),card.play_effects[-1]])
        assert not supports_current_hp_projection(altered,current)
