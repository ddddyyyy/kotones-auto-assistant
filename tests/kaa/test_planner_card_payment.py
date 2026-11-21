from dataclasses import replace
import pytest
from kaa.db.skill_card import SkillCard
from kaa.tasks.produce.new.play_cards.planner.card_payment import CardPaymentObservation,valid_card_payment
from kaa.tasks.produce.new.play_cards.planner.costs import observed_play_cost,play_cost
from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play,can_afford_play
from kaa.idol_road.current_hp_projection import supports_current_hp_projection
from tests.kaa.test_idol_road_hp_safety import state


def current(hp=46,genki=2,cost=2):
    card=SkillCard.from_id('p_card-00-acc-0_002')
    proof=CardPaymentObservation('frame-2',card._id,0,(300,1060,45,40),cost,1.,1.,1.,True)
    value=replace(state(turn=2,hp=hp),genki=genki,hand_card_ids=(card._id,),playable_count=2,
        observed_effects=frozenset({'stamina_consumption_modifier'}),observed_card_payments=(proof,))
    return value,card,proof


def test_native_fee_is_current_only_and_spends_genki_before_hp():
    value,card,proof=current()
    assert valid_card_payment(proof,'frame-2')
    assert observed_play_cost(value,card)==(2,0) and play_cost(value,card)==(2,0)
    assert supports_current_hp_projection(card,value) and can_afford_play(value,card) is True
    after=simulate_play(value,card)
    assert (after.hp,after.genki,after.playable_count)==(46,0,1)
    assert after.hp_known and after.genki_known
    assert observed_play_cost(after,card) is None
    assert card.stamina==0


@pytest.mark.parametrize('hp,genki,legal',[(0,2,True),(1,0,False),(2,0,True)])
def test_zero_hp_is_legal_but_insufficient_payment_is_not(hp,genki,legal):
    value,card,_=current(hp,genki)
    assert can_afford_play(value,card) is legal


@pytest.mark.parametrize('change',[{'frame_id':'stale'},{'green':True},{'green':3},{'unique':False},
    {'score':.96},{'coverage':.94},{'green_coverage':.59},{'upgrade':1},{'rect':(700,1060,45,40)}])
def test_incomplete_or_other_frame_evidence_cannot_override_cost(change):
    value,card,proof=current()
    altered=replace(proof,**change)
    assert not valid_card_payment(altered,'frame-2')
    assert observed_play_cost(replace(value,observed_card_payments=(altered,)),card) is None


def test_duplicate_sleep_requires_every_copy_and_agreeing_fee():
    value,card,proof=current()
    value=replace(value,hand_card_ids=(card._id,card._id))
    assert observed_play_cost(value,card) is None
    second=replace(proof,rect=(408,1060,45,40))
    assert observed_play_cost(replace(value,observed_card_payments=(proof,second)),card)==(2,0)
    assert observed_play_cost(replace(value,observed_card_payments=(proof,replace(second,green=0))),card) is None
    assert observed_play_cost(replace(value,observed_card_payments=(proof,proof)),card) is None
