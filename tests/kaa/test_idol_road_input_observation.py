"""Single-input scheduling contracts; no device or OCR execution."""
from dataclasses import replace
import pytest
from kaa.idol_road.input_observation import PendingInputObservation, InputObservationError
from kaa.idol_road.page import PageSnapshot
from kaa.idol_road.models import Observation, Scene, Action
from kaa.idol_road.battle_log import BattleLogBinding
from kaa.idol_road.card_probe import pending_card_permit, assess_card_log
from kaa.idol_road.skip_probe import pending_skip_permit
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from tests.kaa.test_idol_road_skip_probe import SESSION, seeded, frame, entry, view


def scenario(kind,*,feedback_only=False,card_remaining=20):
    rounds, seed, inputs = seeded()
    now = [0.]
    inputs.clock = lambda: now[0]
    inputs.begin(kind, seed, 'card' if kind == 'card' else None)
    if kind == 'card':
        inputs.bind_card_state(BattleState(20, 50, 0, hand_size=2, hand_card_ids=('card', 'other'),
            absolute_turn_observation=seed.absolute_turn), 'card')
    permit = pending_card_permit(SESSION, inputs.snapshot(), now=0) or pending_skip_permit(SESSION, inputs.snapshot(), now=0)
    binding = BattleLogBinding(clock=inputs.clock)
    binding.update_inputs(1, True, card_probe=permit if kind == 'card' else None,
        skip_probe=permit if kind == 'skip' else None)
    remaining = card_remaining if kind == 'card' else 19
    for name in ('a', 'b'):
        binding.observe_hud(frame(name, remaining, 49), entry(name))
    policy = PendingInputObservation(SESSION, inputs.snapshot(), deadline=100, clock=inputs.clock)
    opened = PageSnapshot(Observation(Scene.BATTLE, 'ttmr', 20, battle_log_available=True),
        battle_frame=frame('b', remaining, 49))
    assert policy.step(opened, inputs.snapshot()).action == Action.INSPECT_BATTLE_LOG
    binding.open(opened.battle_frame, entry('b'), input_sequence=1,
        card_probe=permit if kind == 'card' else None, skip_probe=permit if kind == 'skip' else None)
    log = view(21-card_remaining if kind == 'card' else 2, remaining)
    inspection = binding.inspect(log, SESSION)
    log_snapshot = PageSnapshot(Observation(Scene.BATTLE_LOG, 'ttmr', 20, battle_log_inspection=inspection))
    assert policy.step(log_snapshot, inputs.snapshot()).action == Action.CLOSE_BATTLE_LOG
    binding.close(inspection)
    raw = frame('back', remaining, 49)
    if feedback_only:
        raw=replace(raw,values=replace(raw.values,detail_mode='feedback_only'))
    returned = binding.observe_hud(raw, entry('back'))
    if kind == 'card':
        resolution = assess_card_log(permit, returned, log, raw, inputs.snapshot(), now=0)
        assert rounds.accept_card(raw, returned, log, resolution, inputs.snapshot(), now=0)
        obs = Observation(Scene.BATTLE, 'ttmr', 20, battle_log_return=returned, battle_card_resolution=resolution)
    else:
        resolution = inputs.finish_skip_log(returned, log, raw)
        assert rounds.accept_skip(raw, returned, resolution, inputs.snapshot())
        obs = Observation(Scene.BATTLE, 'ttmr', 20, battle_log_return=returned, battle_skip_resolution=resolution)
    return policy, inputs, now, PageSnapshot(obs, battle_frame=rounds.bind(raw, inputs.snapshot())), log_snapshot


@pytest.mark.parametrize('kind', ['card', 'skip'])
def test_current_owned_return_is_ready_without_selecting_a_new_action(kind):
    policy, inputs, _, snapshot, _ = scenario(kind)
    decision = policy.step(snapshot, inputs.snapshot())
    assert decision.action == Action.READY
    assert (inputs.pending is not None) == (kind == 'card')
    assert policy.deadline == 20


@pytest.mark.parametrize('kind',['card','skip'])
@pytest.mark.parametrize('scene',[Scene.RESULT,Scene.JUDGING])
def test_owned_terminal_returns_control_without_claiming_card_effects(kind,scene):
    policy,inputs,_,_,_=scenario(kind)
    before=inputs.snapshot()
    terminal=PageSnapshot(Observation(scene,'ttmr',20))
    decision=policy.step(terminal,before)
    assert decision.action==Action.READY and decision.reason=='pending_input_owned_battle_finished'
    assert inputs.snapshot()==before


def test_foreign_terminal_cannot_be_used_to_finish_owned_input():
    policy,inputs,_,_,_=scenario('card')
    terminal=PageSnapshot(Observation(Scene.RESULT,'hski',20))
    assert policy.step(terminal,inputs.snapshot()).reason=='pending_input_identity_changed'


@pytest.mark.parametrize('kind', ['card', 'skip'])
def test_late_owned_return_never_releases_observation_wait(kind):
    policy, inputs, now, snapshot, _ = scenario(kind)
    now[0] = 20.
    assert policy.step(snapshot, inputs.snapshot()).reason == 'pending_input_observation_timeout'
    now[0] = 1.
    assert policy.step(snapshot, inputs.snapshot()).action == Action.STOP


@pytest.mark.parametrize('change', [
    {'sequence': 2}, {'battle_id': 'other'}, {'expired': True},
])
def test_input_identity_and_sequence_changes_stop(change):
    policy, inputs, _, snapshot, _ = scenario('card')
    assert policy.step(snapshot, replace(inputs.snapshot(), **change)).action == Action.STOP


def test_cleared_card_ledger_cannot_be_adopted_as_own_feedback():
    policy, inputs, _, snapshot, _ = scenario('card')
    assert policy.step(snapshot, replace(inputs.snapshot(), pending=None)).reason == 'pending_input_changed'


def test_changed_log_request_even_with_same_permit_is_rejected():
    policy, inputs, _, _, snapshot = scenario('card')
    inspection = snapshot.observation.battle_log_inspection
    request = replace(inspection.request, request_id='other')
    snapshot.observation = replace(snapshot.observation, battle_log_inspection=replace(inspection, request=request))
    assert policy.step(snapshot, inputs.snapshot()).reason == 'pending_input_log_unowned'


def test_unknown_and_pending_return_frames_do_not_reopen_the_log():
    policy, inputs, _, _, _ = scenario('card')
    for obs in (Observation(Scene.UNKNOWN), Observation(Scene.BATTLE, battle_log_pending=True)):
        assert policy.step(PageSnapshot(obs), inputs.snapshot()).action == Action.WAIT
    assert policy.log_requested


def test_no_pending_origin_cannot_create_an_observer():
    _, _, inputs = seeded()
    with pytest.raises(InputObservationError):
        PendingInputObservation(SESSION, inputs.snapshot(), deadline=100, clock=lambda: 0.)


def test_sampling_is_bounded_even_with_frozen_clock():
    policy, inputs, _, _, _ = scenario('card')
    for _ in range(48):
        assert policy.step(PageSnapshot(Observation(Scene.UNKNOWN)), inputs.snapshot()).action == Action.WAIT
    assert policy.step(PageSnapshot(Observation(Scene.UNKNOWN)), inputs.snapshot()).action == Action.STOP


def test_wrong_or_unconfirmed_return_stops_instead_of_opening_a_second_window():
    policy, inputs, _, snapshot, _ = scenario('card')
    snapshot.observation = replace(snapshot.observation, battle_card_resolution=None)
    assert policy.step(snapshot, inputs.snapshot()).reason == 'pending_input_return_unconfirmed'


def test_outer_deadline_can_shorten_but_not_extend_input_deadline():
    _, inputs, now, snapshot, _ = scenario('card')
    policy = PendingInputObservation(SESSION, inputs.snapshot(), deadline=2., clock=inputs.clock)
    now[0] = 2.
    assert policy.step(snapshot, inputs.snapshot()).reason == 'pending_input_observation_timeout'


@pytest.mark.parametrize('identity', [{'character_id': 'other'}, {'stage': 19}])
def test_explicit_wrong_identity_in_unknown_frame_is_not_waited_through(identity):
    policy, inputs, _, _, _ = scenario('card')
    snapshot = PageSnapshot(Observation(Scene.UNKNOWN, **identity))
    assert policy.step(snapshot, inputs.snapshot()).reason == 'pending_input_identity_changed'


def test_same_permit_and_request_id_with_different_opening_frame_is_rejected():
    policy, inputs, _, _, snapshot = scenario('card')
    inspection = snapshot.observation.battle_log_inspection
    request = replace(inspection.request, opening_frame_id='other')
    snapshot.observation = replace(snapshot.observation, battle_log_inspection=replace(inspection, request=request))
    assert policy.step(snapshot, inputs.snapshot()).reason == 'pending_input_log_unowned'


def test_return_source_and_empty_request_id_cannot_be_adopted():
    policy, inputs, _, snapshot, _ = scenario('card')
    snapshot.observation = replace(snapshot.observation,
        battle_log_return=replace(snapshot.observation.battle_log_return, source='historical'))
    assert policy.step(snapshot, inputs.snapshot()).reason == 'pending_input_return_unowned'
    policy, inputs, _, _, snapshot = scenario('card')
    inspection = snapshot.observation.battle_log_inspection
    policy.request_id = None
    snapshot.observation = replace(snapshot.observation,
        battle_log_inspection=replace(inspection, request=replace(inspection.request, request_id='')))
    assert policy.step(snapshot, inputs.snapshot()).reason == 'pending_input_log_unowned'
