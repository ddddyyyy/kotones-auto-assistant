from dataclasses import replace
import pytest
from kaa.battle_turn import AbsoluteTurnObservation
from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame
from kaa.idol_road.battle_inputs import BattleInputState,PendingBattleInput
from kaa.idol_road.log_hand import LogHandCard,confirm_log_hand
from kaa.idol_road.hud_feedback import HudFeedbackWindow

SESSION=BattleObservationSession('battle','ttmr',20,20,'confirmed_formation_start_request','variable_budget')

def frame(name,remaining=20,hp=48):
    return bind_battle_frame(SESSION,BattleFrameValues(name,True,remaining,hp,detail_mode='core_only'))

def hand(name,ids):
    return confirm_log_hand(name,len(ids),tuple(LogHandCard(card,0,True,(100+i*100,900,50,100),False)
        for i,card in enumerate(ids)),empty_confirmed=not ids)

def scenario(kind='card',ids=('card','other')):
    before=frame('before')
    before=replace(before,values=replace(before.values,detail_mode='complete'),
        observation=replace(before.observation,turn=1,turn_source='owned_initial_log_return'),
        absolute_turn=AbsoluteTurnObservation('battle','before',1,20))
    pending=PendingBattleInput(1,kind,before,1,'card' if kind=='card' else None,10.)
    inputs=BattleInputState('battle',1,pending,False)
    return inputs,HudFeedbackWindow(inputs,hand('before',ids))

def test_card_removal_produces_evidence_without_clearing_ledger_or_assigning_turn():
    inputs,window=scenario()
    assert window.observe(inputs,frame('one'),hand('one',('other',)),now=11) is None
    result=window.observe(inputs,frame('two'),hand('two',('other',)),now=12)
    assert result.change=='requested_card_removed'
    assert result.frame.absolute_turn is None and inputs.pending is not None

def test_redraw_is_distinct_from_confirmed_card_removal():
    inputs,window=scenario()
    window.observe(inputs,frame('one'),hand('one',('new','drawn')),now=11)
    result=window.observe(inputs,frame('two'),hand('two',('new','drawn')),now=12)
    assert result.change=='hand_changed' and inputs.pending is not None

def test_duplicate_redraw_or_resource_change_alone_does_not_confirm_card():
    inputs,window=scenario()
    for name in ('one','two','three'):
        assert window.observe(inputs,frame(name,hp=40),hand(name,('card','other')),now=12) is None

def test_exact_duplicate_counts_are_preserved():
    inputs,window=scenario(ids=('card','card','other'))
    window.observe(inputs,frame('one'),hand('one',('card','other')),now=11)
    assert window.observe(inputs,frame('two'),hand('two',('card','other')),now=12).change=='requested_card_removed'


def test_mixed_upgrade_duplicates_do_not_claim_which_requested_card_left():
    inputs,_=scenario(ids=('card','card','other'))
    original=hand('before',('card','card','other'))
    cards=(replace(original.cards[0],upgrade_count=1),*original.cards[1:])
    original=confirm_log_hand('before',3,cards)
    window=HudFeedbackWindow(inputs,original)
    window.observe(inputs,frame('one'),hand('one',('card','other')),now=11)
    assert window.observe(inputs,frame('two'),hand('two',('card','other')),now=12).change=='hand_changed'


def test_selected_upgrade_can_distinguish_mixed_duplicate_cards():
    inputs,_=scenario(ids=('card','card','other'))
    inputs=replace(inputs,pending=replace(inputs.pending,card_upgrade=1))
    original=hand('before',('card','card','other'))
    original=confirm_log_hand('before',3,(replace(original.cards[0],upgrade_count=1),*original.cards[1:]))
    window=HudFeedbackWindow(inputs,original)
    window.observe(inputs,frame('one'),hand('one',('card','other')),now=11)
    assert window.observe(inputs,frame('two'),hand('two',('card','other')),now=12).change=='requested_card_removed'


@pytest.mark.parametrize('was_available,expected',[(False,'requested_card_removed'),(True,'hand_changed')])
def test_unavailable_neighbour_exposing_upgrade_does_not_obscure_unique_requested_card(was_available,expected):
    inputs,_=scenario()
    original=hand('before',('card','other'))
    original=confirm_log_hand('before',2,(original.cards[0],replace(original.cards[1],available=was_available)))
    window=HudFeedbackWindow(inputs,original)
    for name in ('one','two'):
        current=hand(name,('other',))
        current=confirm_log_hand(name,1,(replace(current.cards[0],upgrade_count=1,available=True),))
        result=window.observe(inputs,frame(name),current,now=12)
    assert result.change==expected


@pytest.mark.parametrize('old_spacing,new_spacing,new_growth,expected',[
    (164,216,False,'requested_card_removed'),
    (164,164,False,'hand_changed'),
    (201,216,False,'hand_changed'),
    (164,216,True,'hand_changed'),
])
def test_reflow_can_expose_a_still_disabled_suffix_only_after_proven_overlap(
        old_spacing,new_spacing,new_growth,expected):
    ids=('left','other','third','card')
    inputs,_=scenario(ids=ids)
    original=confirm_log_hand('before',4,tuple(LogHandCard(key,0,key=='card',
        (30+index*old_spacing,935,57,148),False) for index,key in enumerate(ids)))
    window=HudFeedbackWindow(inputs,original)
    for name in ('one','two'):
        current=confirm_log_hand(name,3,tuple(LogHandCard(key,int(key=='other'),key=='third',
            (80+index*new_spacing,935,57,148),new_growth if key=='other' else False)
            for index,key in enumerate(ids[:-1])))
        result=window.observe(inputs,frame(name),current,now=12)
    assert result.change==expected


def test_unchanged_available_copy_cannot_vouch_for_disabled_duplicate_version_change():
    inputs,_=scenario(ids=('card','other','other','third'))
    original=confirm_log_hand('before',4,tuple(LogHandCard(key,int(index==2),index!=1,
        (30+index*201,935,57,148),False) for index,key in enumerate(('card','other','other','third'))))
    window=HudFeedbackWindow(inputs,original)
    for name in ('one','two'):
        current=confirm_log_hand(name,3,tuple(LogHandCard(key,int(key=='other'),index!=0,
            (80+index*216,935,57,148),False) for index,key in enumerate(('other','other','third'))))
        result=window.observe(inputs,frame(name),current,now=12)
    assert result.change=='hand_changed'


def test_unchanged_duplicate_trouble_cards_do_not_obscure_another_neighbour_unlocking():
    inputs,_=scenario(ids=('sleep','card','other','sleep'))
    original=hand('before',('sleep','card','other','sleep'))
    original=confirm_log_hand('before',4,tuple(replace(card,available=card.card_id=='card') for card in original.cards))
    window=HudFeedbackWindow(inputs,original)
    for name in ('one','two'):
        current=hand(name,('sleep','other','sleep'))
        current=confirm_log_hand(name,3,tuple(replace(card,upgrade_count=1 if card.card_id=='other' else 0,
            available=card.card_id=='other') for card in current.cards))
        result=window.observe(inputs,frame(name),current,now=12)
    assert result.change=='requested_card_removed'


def test_a_playable_but_overlapped_neighbour_can_reveal_its_version_after_unique_removal():
    inputs,_=scenario(ids=('sleep','card','other','sleep'))
    original=hand('before',('sleep','card','other','sleep'))
    original=confirm_log_hand('before',4,tuple(replace(card,rect=(10+i*160,936,57,148)) for i,card in enumerate(original.cards)))
    window=HudFeedbackWindow(inputs,original)
    for name in ('one','two'):
        current=hand(name,('sleep','other','sleep'))
        current=confirm_log_hand(name,3,tuple(replace(card,upgrade_count=1 if card.card_id=='other' else 0) for card in current.cards))
        result=window.observe(inputs,frame(name),current,now=12)
    assert result.change=='requested_card_removed'

@pytest.mark.parametrize('remaining',[19,21])
def test_skip_observes_remaining_change_without_fixed_budget_arithmetic(remaining):
    inputs,window=scenario('skip')
    window.observe(inputs,frame('one',remaining),hand('one',('next',)),now=11)
    result=window.observe(inputs,frame('two',remaining),hand('two',('next',)),now=12)
    assert result.change=='skip_remaining_changed' and result.frame.absolute_turn is None

def test_same_capture_cannot_count_as_two_frames():
    inputs,window=scenario()
    for _ in range(3):
        assert window.observe(inputs,frame('one'),hand('one',('other',)),now=11) is None

def test_unknown_current_hp_interrupts_stability_and_is_not_borrowed():
    inputs,window=scenario()
    window.observe(inputs,frame('one'),hand('one',('other',)),now=11)
    window.observe(inputs,frame('unknown',hp=None),hand('unknown',('other',)),now=12)
    assert window.observe(inputs,frame('two'),hand('two',('other',)),now=13) is None
    assert window.observe(inputs,frame('three'),hand('three',('other',)),now=14) is not None

@pytest.mark.parametrize('change',['sequence','pending','expired','deadline'])
def test_original_request_and_deadline_are_mandatory(change):
    inputs,window=scenario()
    other=replace(inputs,**({'sequence':2} if change=='sequence' else
        {'pending':None} if change=='pending' else {'expired':True} if change=='expired' else {}))
    with pytest.raises(ValueError):
        window.observe(other,frame('one'),hand('one',('other',)),now=30 if change=='deadline' else 11)

def test_foreign_session_does_not_produce_evidence():
    inputs,window=scenario()
    foreign=replace(frame('one'),session=replace(SESSION,battle_id='other'))
    with pytest.raises(ValueError):window.observe(inputs,foreign,hand('one',('other',)),now=11)
