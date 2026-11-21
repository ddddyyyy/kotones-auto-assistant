from dataclasses import replace
import pytest
from kaa.idol_road.log_hand import LogHandCard,fingerprint_log_hand


def cards():
    return (LogHandCard('a',1,True,(20,900,90,180),False),
            LogHandCard('b',0,False,(230,900,90,180),False))


def test_fingerprint_uses_current_visible_cards_and_slot_order_not_dynamic_background():
    first,second=cards()
    assert fingerprint_log_hand(2,(first,second))==fingerprint_log_hand(2,(second,first))
    assert fingerprint_log_hand(2,cards()) is not None


@pytest.mark.parametrize('changes',[{'card_id':'other'},{'upgrade_count':0},{'available':False},
    {'rect':(21,900,90,180)},{'growth_marker':True}])
def test_observed_hand_change_invalidates_fingerprint(changes):
    first,second=cards()
    assert fingerprint_log_hand(2,cards())!=fingerprint_log_hand(2,(replace(first,**changes),second))


@pytest.mark.parametrize('changes',[{'card_id':''},{'upgrade_count':-1},{'available':None},
    {'rect':(-1,900,90,180)},{'rect':(20,900,90,400)},{'growth_marker':None}])
def test_unknown_or_invalid_card_cannot_supply_stable_hand(changes):
    first,second=cards()
    assert fingerprint_log_hand(2,(replace(first,**changes),second)) is None


def test_count_mismatch_duplicate_slot_and_unknown_empty_are_rejected():
    first,_=cards()
    assert fingerprint_log_hand(3,cards()) is None
    assert fingerprint_log_hand(2,(first,first)) is None
    assert fingerprint_log_hand(True,(first,)) is None
    assert fingerprint_log_hand(0,()) is None
    assert fingerprint_log_hand(0,(),empty_confirmed=True) is not None
