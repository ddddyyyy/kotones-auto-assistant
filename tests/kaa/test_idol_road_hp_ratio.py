from dataclasses import replace
import pytest
from kaa.idol_road.hp_ratio import HpRatioRead, confirm_hp_ratio
from kaa.idol_road.battle_observation import BattleFrameValues, bind_battle_frame, create_battle_session
from kaa.idol_road.models import Observation, Scene


def ratio(left='44/48', right=None, frame='frame', confidence=.99):
    return confirm_hp_ratio(frame, (329,193,47,25), .99, (
        HpRatioRead(frame, 'original', left, confidence),
        HpRatioRead(frame, 'grayscale', left if right is None else right, confidence)))


@pytest.mark.parametrize('text,hp', [('0/48',0),('44/48',44),('48/48',48),(' 44 / 48 ',44)])
def test_explicit_pair_resolves(text,hp):
    observed=ratio(text)
    assert observed.status=='resolved' and (observed.hp,observed.max_hp)==(hp,48)


@pytest.mark.parametrize('text', ['44','44/0','44/048','44/48extra','-1/48','44/48/48',''])
def test_incomplete_or_malformed_pair_unknown(text):
    assert ratio(text).status=='unknown'


@pytest.mark.parametrize('left,right', [('49/48','49/48'),('44/48','44/49'),('44/48','43/48')])
def test_conflicting_pair_is_not_selected(left,right):
    assert ratio(left,right).status=='conflict'


@pytest.mark.parametrize('confidence', [.97,float('nan'),1.1])
def test_uncertain_pair_stays_unknown(confidence):
    assert ratio(confidence=confidence).status=='unknown'


def test_label_and_same_frame_both_required():
    value=ratio()
    assert confirm_hp_ratio('frame',value.label_rect,.96,value.reads).status=='unknown'
    assert confirm_hp_ratio('new',value.label_rect,.99,value.reads).status=='unknown'


def bound(**changes):
    session=create_battle_session(Observation(Scene.CONFIRM,'ttmr',16,stars=0,formation_ready=True,total_turns=10))
    return bind_battle_frame(session,replace(BattleFrameValues('frame',True,4,44,hp_ratio=ratio()),**changes)).observation


def test_pair_binds_same_frame_and_is_not_carried_to_next_frame():
    assert (bound().hp,bound().max_hp)==(44,48)
    assert bound(frame_id='next',hp_ratio=None).max_hp is None


@pytest.mark.parametrize('changes', [{'hp':43},{'hp_ratio':ratio(frame='old')},
    {'hp_ratio':replace(ratio(),max_hp=99)},{'hp_ratio':ratio('44/48','43/48')}])
def test_hud_stale_or_forged_pair_conflicts(changes):
    state=bound(**changes)
    assert state.hp is None and state.max_hp is None
    assert {'hp','max_hp'} <= set(state.conflicted_fields)


@pytest.mark.parametrize('preview',['selection_preview','unconfirmed_overlay'])
def test_preview_drops_even_explicit_pair(preview):
    state=bound(resource_state=preview)
    assert state.hp is None and state.max_hp is None and state.turn is None
    assert state.source=='unconfirmed_resource_preview'
