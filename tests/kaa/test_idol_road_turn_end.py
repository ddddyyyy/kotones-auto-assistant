from dataclasses import replace
import pytest
from kaa.idol_road.turn_end import HandCardState, UnplayableTurnGate, hand_is_blocked


def cards():
    return (HandCardState('mental-a', False, (10, 20, 30, 40)),
            HandCardState('mental-b', False, (50, 20, 30, 40)))


def test_complete_recognized_disabled_hand_is_required():
    assert hand_is_blocked(2, cards())
    for count in (None, 0, 1, 3, True):
        assert not hand_is_blocked(count, cards())


@pytest.mark.parametrize('change', [
    {'card_id': None}, {'available': True}, {'available': None},
    {'rect': None}, {'rect': (10, 20, 0, 40)},
])
def test_unknown_card_or_playable_card_or_invalid_rect_cannot_end_turn(change):
    first, second = cards()
    assert not hand_is_blocked(2, (replace(first, **change), second))


def test_duplicate_slots_do_not_prove_a_complete_hand():
    first = cards()[0]
    assert not hand_is_blocked(2, (first, first))


def test_only_blocked_hand_with_enabled_skip_requests_end_turn():
    gate = UnplayableTurnGate()
    for blocked, enabled in ((False, True), (True, False)):
        assert gate.step('battle', 4, blocked, enabled) == 'none'
    assert gate.step('battle', 4, True, True) == 'click'


def test_request_is_not_repeated_on_stale_observations():
    gate = UnplayableTurnGate()
    assert gate.step('battle', 4, True, True) == 'click'
    assert gate.step('battle', 4, True, True) == 'wait'
    assert gate.step('battle', 4, False, False) == 'wait'
    assert gate.step('battle', 4, True, True) == 'stop'


def test_new_turn_does_not_inherit_previous_disabled_hand():
    gate = UnplayableTurnGate()
    assert gate.step('battle', 4, True, True) == 'click'
    assert gate.step('battle', 5, False, True) == 'none'
    assert gate.step('battle', 5, True, True) == 'click'
def test_planned_skip_blocks_even_newly_visible_playable_cards_until_turn_advances():
    from kaa.idol_road.turn_end import PlannedTurnWait
    now = [0.]
    gate = PlannedTurnWait(clock=lambda: now[0])
    assert gate.check(2) == 'continue'
    gate.requested(2)
    assert gate.check(2) == 'wait'
    assert gate.check(None) == 'wait'
    now[0] = 19.9
    assert gate.check(2) == 'wait'
    assert gate.check(3) == 'continue'
    assert gate.pending is None
    gate.requested(5)
    assert gate.check(4) == 'stop'
    now[0] = 40
    assert gate.check(None) == 'stop'


@pytest.mark.parametrize('elapsed', [20., 20.001, 30., -0.001])
def test_late_or_clock_reversed_new_turn_cannot_release_skip_wait(elapsed):
    from kaa.idol_road.turn_end import PlannedTurnWait
    now = [100.]
    gate = PlannedTurnWait(clock=lambda: now[0])
    gate.requested(1)
    now[0] += elapsed
    assert gate.check(2) == 'stop'
    assert gate.pending == (1, 100.)
    now[0] = 101.
    assert gate.check(2) == 'stop'
    with pytest.raises(ValueError):
        gate.requested(2)


def test_skip_wait_uses_original_input_start_instead_of_registration_time():
    from kaa.idol_road.turn_end import PlannedTurnWait
    now = [105.]
    gate = PlannedTurnWait(clock=lambda: now[0])
    gate.requested(1, started=100.)
    now[0] = 119.999
    assert gate.check(None) == 'wait'
    now[0] = 120.
    assert gate.check(2) == 'stop'


@pytest.mark.parametrize('origin', [True, float('inf'), float('nan'), 106., 85.])
def test_skip_wait_rejects_invalid_or_expired_input_start(origin):
    from kaa.idol_road.turn_end import PlannedTurnWait
    gate = PlannedTurnWait(clock=lambda: 105.)
    with pytest.raises(ValueError):
        gate.requested(1, started=origin)
    assert gate.pending is None


def test_round_regression_is_terminal_even_if_new_round_appears_later():
    from kaa.idol_road.turn_end import PlannedTurnWait
    gate = PlannedTurnWait(clock=lambda: 100.)
    gate.requested(4)
    assert gate.check(3) == 'stop'
    assert gate.check(5) == 'stop'
