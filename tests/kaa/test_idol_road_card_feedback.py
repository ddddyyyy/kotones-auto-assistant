from dataclasses import replace
from types import SimpleNamespace
import pytest
from kaa.idol_road.card_feedback import settle_card_hand_feedback
from kaa.idol_road.log_hand import LogHandCard, confirm_log_hand
from kaa.idol_road.input_policy import InputBoundPolicy
from kaa.tasks.produce.new.play_cards.planner.feedback import PendingPlay
from tests.kaa.test_idol_road_input_observation import scenario


def current_feedback(*,feedback_only=False,advance=False):
    _, inputs, now, snapshot, _ = scenario('card',feedback_only=feedback_only,card_remaining=19 if advance else 20)
    hand = confirm_log_hand(snapshot.battle_frame.values.frame_id, 1,
        (LogHandCard('other', 0, True, (100, 900, 100, 100), False),))
    returned = snapshot.observation.battle_log_return
    returned = replace(returned, request=replace(returned.request, hand_fingerprint=hand.fingerprint))
    snapshot.log_hand = hand
    snapshot.observation = replace(snapshot.observation, battle_log_hand=hand, battle_log_return=returned)
    observed = []
    policy = InputBoundPolicy(inputs, lambda: snapshot.battle_frame,
        delegate=SimpleNamespace(feedback=lambda pending, feedback, state: observed.append(state)))
    pending = PendingPlay('card', inputs.pending.before, 0, 0)
    return policy, now, snapshot, pending, observed


def test_complete_owned_hand_can_settle_before_planning_ocr():
    policy, _, snapshot, pending, observed = current_feedback()
    result = settle_card_hand_feedback(policy, snapshot, pending)
    assert result.feedback.evidence == 'card_left_hand' and policy.inputs.pending is None
    assert result.frame_id == snapshot.battle_frame.values.frame_id
    assert result.hand_fingerprint == snapshot.log_hand.fingerprint
    assert observed[0].genki_known is False and observed[0].current_score is None
    assert observed[0].hp == snapshot.battle_frame.observation.hp
    assert observed[0].hand_card_ids == ('other',)


@pytest.mark.parametrize('now_value,confirmed', [(19.999, True), (20., False), (21., False)])
def test_original_deadline_still_controls_hand_only_settlement(now_value, confirmed):
    policy, now, snapshot, pending, _ = current_feedback()
    now[0] = now_value
    assert (settle_card_hand_feedback(policy, snapshot, pending) is not None) == confirmed
    assert (policy.inputs.pending is None) == confirmed


@pytest.mark.parametrize('change', ['missing', 'wrong_frame', 'bad_fingerprint', 'different_return_fingerprint', 'missing_observation'])
def test_hand_provenance_is_required(change):
    policy, _, snapshot, pending, _ = current_feedback()
    if change == 'missing':
        snapshot.log_hand = None
    elif change == 'missing_observation':
        snapshot.observation = replace(snapshot.observation, battle_log_hand=None)
    elif change == 'different_return_fingerprint':
        returned = snapshot.observation.battle_log_return
        snapshot.observation = replace(snapshot.observation,
            battle_log_return=replace(returned, request=replace(returned.request, hand_fingerprint='a' * 64)))
    else:
        snapshot.log_hand = replace(snapshot.log_hand, **({'frame_id': 'old'} if change == 'wrong_frame' else {'fingerprint': 'a' * 64}))
        snapshot.observation = replace(snapshot.observation, battle_log_hand=snapshot.log_hand)
    assert settle_card_hand_feedback(policy, snapshot, pending) is None
    assert policy.inputs.pending is not None


def test_resource_change_without_card_leaving_does_not_use_hand_only_path():
    policy, _, snapshot, pending, _ = current_feedback()
    hand = confirm_log_hand(snapshot.log_hand.frame_id, 2, (
        LogHandCard('card', 0, True, (100, 900, 100, 100), False),
        LogHandCard('other', 0, True, (300, 900, 100, 100), False)))
    snapshot.log_hand = hand
    returned = snapshot.observation.battle_log_return
    snapshot.observation = replace(snapshot.observation, battle_log_hand=hand,
        battle_log_return=replace(returned, request=replace(returned.request, hand_fingerprint=hand.fingerprint)))
    assert settle_card_hand_feedback(policy, snapshot, pending) is None
    assert policy.inputs.pending is not None


def test_stale_selected_card_before_cannot_settle_a_different_pending_play():
    policy, _, snapshot, pending, _ = current_feedback()
    assert settle_card_hand_feedback(policy, snapshot, replace(pending, card_id='other')) is None
    assert policy.inputs.pending is not None


def test_relabelled_return_source_cannot_settle_even_with_matching_frame_and_hand():
    policy, _, snapshot, pending, _ = current_feedback()
    snapshot.observation = replace(snapshot.observation,
        battle_log_return=replace(snapshot.observation.battle_log_return, source='historical'))
    assert settle_card_hand_feedback(policy, snapshot, pending) is None
    assert policy.inputs.pending is not None


@pytest.mark.parametrize('now_value,confirmed',[(19.999,True),(20.,False)])
def test_feedback_only_owned_hand_settles_without_becoming_planning_state(now_value,confirmed):
    from kaa.idol_road.battle_observation import current_absolute_turn,current_feedback_turn
    from kaa.idol_road.planner_context import IdolRoadBattleContext
    policy,now,snapshot,pending,_=current_feedback(feedback_only=True)
    frame=snapshot.battle_frame
    assert current_absolute_turn(frame) is None and current_feedback_turn(frame).turn==1
    assert frame.values.detail_mode=='feedback_only' and frame.values.concentration is None
    now[0]=now_value
    result=settle_card_hand_feedback(policy,snapshot,pending)
    assert (result is not None)==confirmed and (policy.inputs.pending is None)==confirmed
    assert not policy.inputs.can_select(frame,pending.before)
    with pytest.raises(ValueError):policy.inputs.begin('card',frame,'next')
    with pytest.raises(ValueError,match='日志探针'):
        IdolRoadBattleContext(None,frame).fetch_exam_current_score()


@pytest.mark.parametrize('source',['owned_initial_log_return','owned_skip_log_return'])
def test_feedback_only_frame_cannot_borrow_a_non_card_round_source(source):
    from kaa.idol_road.battle_observation import current_feedback_turn
    policy,_,snapshot,pending,_=current_feedback(feedback_only=True)
    snapshot.battle_frame=replace(snapshot.battle_frame,
        observation=replace(snapshot.battle_frame.observation,turn_source=source))
    assert current_feedback_turn(snapshot.battle_frame) is None
    assert settle_card_hand_feedback(policy,snapshot,pending) is None
    assert policy.inputs.pending is not None


def test_feedback_only_does_not_allow_resource_change_to_release_the_request():
    from kaa.idol_road.battle_observation import current_feedback_turn
    from kaa.tasks.produce.new.play_cards.planner.state import BattleState
    from kaa.tasks.produce.new.play_cards.planner.feedback import PlayFeedback,PlayFeedbackStatus
    policy,_,snapshot,pending,_=current_feedback(feedback_only=True)
    frame=snapshot.battle_frame
    state=BattleState(20,49,0,absolute_turn_observation=current_feedback_turn(frame))
    assert not policy.inputs.finish_card(pending,
        PlayFeedback(PlayFeedbackStatus.CONFIRMED,'resource_changed'),state,frame)
    assert policy.inputs.pending is not None


@pytest.mark.parametrize('clock',[19.999,20.])
def test_owned_round_advance_can_release_wait_without_confirming_card(clock):
    policy,now,snapshot,pending,observed=current_feedback(feedback_only=True,advance=True)
    now[0]=clock
    assert settle_card_hand_feedback(policy,snapshot,pending) is None
    receipt=settle_card_hand_feedback(policy,snapshot,pending,allow_round_advance=True)
    assert (receipt is not None)==(clock<20)
    assert (policy.inputs.pending is None)==(clock<20)
    if receipt:
        assert receipt.feedback.status.value=='unconfirmed'
        assert receipt.feedback.evidence=='turn_advanced_without_play_observation'
        assert not policy.inputs.can_select(snapshot.battle_frame,observed[0])
