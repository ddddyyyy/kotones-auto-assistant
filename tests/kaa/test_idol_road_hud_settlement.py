from dataclasses import replace
import pytest
from kaa.idol_road.battle_inputs import BattleInputs
from kaa.idol_road.hud_feedback import HudFeedbackWindow
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from tests.kaa.test_idol_road_hud_feedback import SESSION,scenario,frame,hand


def owned_card(after=('other',)):
    origin,_=scenario()
    now=[12.]
    ledger=BattleInputs(SESSION,clock=lambda:now[0])
    # Bind a real ledger request with the current absolute origin frame.
    ledger.begin('card',origin.pending.frame,'card')
    state=BattleState(20,48,0,is_exam=True,hand_size=2,hand_card_ids=('card','other'),
        absolute_turn_observation=origin.pending.frame.absolute_turn)
    ledger.bind_card_state(state,'card',category='ProduceCardCategory_MentalSkill')
    window=HudFeedbackWindow(ledger.snapshot(),hand('before',('card','other')))
    window.observe(ledger.snapshot(),frame('one'),hand('one',after),now=13)
    evidence=window.observe(ledger.snapshot(),frame('two'),hand('two',after),now=14)
    now[0]=14
    return ledger,now,evidence


def test_card_can_settle_without_log_but_returned_frame_still_cannot_select():
    ledger,_,evidence=owned_card()
    assert ledger.finish_hud_card(evidence)
    assert ledger.pending is None and ledger.last_owned_card_play is None
    assert ledger.previous_round_card is None
    assert ledger._last_owned_card_play.source=='owned_hud_card_left_hand'
    assert ledger._last_owned_card_play.turn==1
    state=BattleState(20,48,0,is_exam=True)
    assert ledger.can_select(evidence.frame,state) is False
    assert ledger.hud_round_pending
    assert ledger.can_select(evidence.pending.frame,evidence.pending.before) is False
    with pytest.raises(ValueError):ledger.begin('skip',evidence.pending.frame)
    assert ledger.finish_hud_card(evidence) is False


def test_redraw_change_is_not_relabelled_as_requested_card_removal():
    ledger,_,evidence=owned_card(('new','drawn'))
    assert evidence.change=='hand_changed'
    assert ledger.finish_hud_card(evidence) is False and ledger.pending is not None


@pytest.mark.parametrize('change',['expired','first_frame','hand','origin','request'])
def test_changed_receipt_or_expired_input_cannot_release_ledger(change):
    ledger,now,evidence=owned_card()
    if change=='expired':now[0]=32
    elif change=='first_frame':evidence=replace(evidence,first_frame=evidence.frame)
    elif change=='hand':evidence=replace(evidence,hand=hand('two',('card','other')))
    elif change=='origin':evidence=replace(evidence,origin_hand=hand('before',('card',)))
    elif change=='request':evidence=replace(evidence,pending=replace(evidence.pending,sequence=9))
    assert ledger.finish_hud_card(evidence) is False and ledger.pending is not None
