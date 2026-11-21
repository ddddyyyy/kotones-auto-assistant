from dataclasses import replace
import pytest
from kaa.db.skill_card import SkillCard
from kaa.idol_road.current_action import assess_current_action
from kaa.idol_road.current_hp_projection import supports_current_hp_projection
from kaa.tasks.produce.new.play_cards.planner.state import ExamBuffState
from tests.kaa.test_idol_road_hp_safety import state


def card():
    base=SkillCard.from_id('p_card-01-sup-3_109')
    return SkillCard.from_asset_id(base._asset_id,1)


def current():
    return replace(state(turn=2,hp=49),genki=8,buffs=ExamBuffState(parameter_buff_turns=6),
        observed_effect_values=(('parameter_buff',6),),hand_size=3,
        hand_card_ids=('sleep','hope','p_card-01-sup-3_109'),playable_count=1)


@pytest.mark.parametrize('upgrade,duration',[(0,1),(1,3),(2,4),(3,5)])
def test_all_audited_redraw_versions_use_their_own_duration_without_predicting_draws(upgrade,duration):
    base=SkillCard.from_id('p_card-01-sup-3_109')
    version=SkillCard.from_asset_id(base._asset_id,upgrade)
    assert version is not None and version.upgrade_count==upgrade
    before=current();result=assess_current_action(before,version)
    assert result.root_effects_supported and result.root_projection_available
    assert result.good_condition_after==6+duration
    assert result.hp_after==49 and result.genki_after==5 and result.playable_count_after==1
    assert result.requires_fresh_hand and 'drawn_card_identity_and_on_move' in result.unresolved_events
    assert not supports_current_hp_projection(version,before)
    assert supports_current_hp_projection(version,before,allow_audited_redraw=True)


@pytest.mark.parametrize('turn',[0,2,6,-1,True])
def test_unaudited_or_malformed_redraw_duration_stays_unsupported(turn):
    original=card();first=original.play_effects[0]
    changed=replace(original,play_effects=[replace(first,produce_exam_effect=replace(first.produce_exam_effect,effect_turn=turn)),*original.play_effects[1:]])
    assert not assess_current_action(current(),changed).root_effects_supported


def test_redraw_exports_only_local_roots_and_keeps_product_gate_closed():
    value=assess_current_action(current(),card())
    assert value.root_projection_available
    assert (value.hp_after,value.genki_after,value.good_condition_after,value.playable_count_after)==(49,5,9,1)
    assert value.requires_fresh_hand and value.unresolved_events
    assert not hasattr(value,'hand_card_ids')
    assert not supports_current_hp_projection(card(),current())


def test_zero_hp_can_pay_redraw_with_known_genki_without_opening_product_gate():
    zero=replace(current(),hp=0,genki=3)
    result=assess_current_action(zero,card())
    assert result.root_projection_available and result.hp_after==0 and result.genki_after==0
    assert not supports_current_hp_projection(card(),zero)
    assert not assess_current_action(replace(zero,genki=2),card()).root_projection_available


def test_redraw_root_support_does_not_invent_payment_under_a_native_modifier():
    from kaa.idol_road.hp_safety import HpSafetyPolicy
    from kaa.idol_road.hp_preparation import HpPreparationBudget
    modified=replace(current(),observed_effects=frozenset({'parameter_buff','stamina_consumption_modifier'}))
    result=assess_current_action(modified,card())
    assert result.root_effects_supported and not result.root_projection_available
    assert result.hp_after is None and result.green_cost is None
    guard=HpSafetyPolicy(20,HpPreparationBudget((),None,False),reserve_mode='prefer',allow_audited_redraw=True)
    assert guard.select(modified,(card(),)).allowed_indices==(0,)
    assert guard.review.candidates[0].projected_hp is None


def test_explicit_current_redraw_gate_needs_exact_roots_and_does_not_make_a_visible_followup():
    from kaa.idol_road.category_priority import supports_visible_followup
    from kaa.idol_road.hp_safety import HpSafetyPolicy
    from tests.kaa.test_idol_road_hp_safety import budget
    current_state=current()
    assert not supports_current_hp_projection(card(),current_state)
    assert supports_current_hp_projection(card(),current_state,allow_audited_redraw=True)
    assert not supports_visible_followup(card(),current_state)
    guard=HpSafetyPolicy(20,budget(),allow_audited_redraw=True)
    assert guard.select(current_state,(card(),)).allowed_indices==(0,)
    assert guard.review.candidates[0].projection_scope=='audited_redraw_roots_only'
    assert guard.review.candidates[0].projected_hp==49
    assert not supports_current_hp_projection(card(),replace(current_state,genki_known=False),allow_audited_redraw=True)
    assert not supports_current_hp_projection(card(),replace(current_state,projected_effect_values=()),allow_audited_redraw=True)


@pytest.mark.parametrize('changes',[{'hp_known':False},{'genki_known':False},{'playable_count':0},
    {'projected_effect_values':()},{'projected_card_growth':(('unknown',1),)}])
def test_uncertain_state_does_not_produce_root_projection(changes):
    assert not assess_current_action(replace(current(),**changes),card()).root_projection_available


@pytest.mark.parametrize('changes',[{'stamina':4},{'cost_value':1},{'_play_produce_exam_trigger_id':'unknown'},
    {'_produce_card_status_enchant_id':'unknown'}])
def test_other_cost_or_trigger_contracts_rejected(changes):
    assert not assess_current_action(current(),replace(card(),**changes)).root_projection_available


def test_unknown_chain_cannot_be_ignored():
    value=card();segment=value.play_effects[1]
    changed=replace(segment,produce_exam_effect=replace(segment.produce_exam_effect,_chain_produce_exam_effect_id='unknown'))
    assert not assess_current_action(current(),replace(value,play_effects=[value.play_effects[0],changed,value.play_effects[2]])).root_projection_available
