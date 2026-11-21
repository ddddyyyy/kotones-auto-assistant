from dataclasses import replace
import pytest

from kaa.idol_road.card_history import OwnedCardPlay,freeze_previous_round


def owned():
    return OwnedCardPlay('battle',3,'mental','ProduceCardCategory_MentalSkill',2,'request','return')


def frozen(play=None,**changes):
    args=dict(battle_id='battle',before_turn=2,current_turn=3,input_sequence=4,frame_id='boundary')
    args.update(changes)
    return freeze_previous_round(owned() if play is None else play,**args)


def test_previous_round_evidence_preserves_card_and_boundary_sources():
    result=frozen()
    assert result.last_play==owned() and result.current_turn==3
    assert result.boundary_input_sequence==4 and result.boundary_frame_id=='boundary'
    assert result.source=='owned_skip_log_return'
    assert not result.last_play.previous_card_condition_confirmed


@pytest.mark.parametrize('changes',[{'battle_id':'other'},{'before_turn':1},{'current_turn':4},
    {'input_sequence':3},{'input_sequence':True},{'frame_id':'return'},{'frame_id':''}])
def test_other_battle_stale_turn_gap_or_wrong_boundary_cannot_freeze(changes):
    assert frozen(**changes) is None


def test_an_older_rounds_last_card_is_not_substituted_for_a_round_with_no_known_play():
    assert frozen(replace(owned(),turn=1)) is None
    assert freeze_previous_round(None,battle_id='battle',before_turn=2,current_turn=3,
                                 input_sequence=4,frame_id='boundary') is None
