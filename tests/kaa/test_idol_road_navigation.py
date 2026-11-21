import pytest

from kaa.idol_road.models import Action, Decision, Observation, Scene, Stage
from kaa.idol_road.navigation import NavigationGuard


def test_invalid_wait_limit_is_rejected():
    with pytest.raises(ValueError):
        NavigationGuard(max_waits=0)


@pytest.mark.parametrize('scene,action', [
    (Scene.HOME, Action.OPEN_CONTEST),
    (Scene.CONTEST_HUB, Action.OPEN_ROAD),
    (Scene.STAGES, Action.SELECT_STAGE),
    (Scene.STAGES, Action.SCAN_STAGES),
    (Scene.CONFIRM, Action.BACK_TO_STAGES),
    (Scene.RESULT, Action.BACK_TO_CHARACTERS),
    (Scene.RESULT, Action.DISMISS_RESULT),
    (Scene.JUDGING, Action.ADVANCE_RESULT),
    (Scene.REWARDS, Action.DISMISS_REWARDS),
    (Scene.REWARD_INFO, Action.CLOSE_INFO),
    (Scene.CONFIRM, Action.OPEN_SUPPORT),
    (Scene.SUPPORT_DETAILS, Action.CLOSE_SUPPORT),
    (Scene.CHARACTERS, Action.SELECT_CHARACTER),
])
def test_page_changing_request_waits_then_stalls(scene, action):
    guard = NavigationGuard()
    observation = Observation(scene, character_selectable=True)
    guard.record(observation, Decision(action, 'test'))
    for _ in range(2):
        assert guard.poll(observation) == 'waiting'
        guard.record(observation, Decision(Action.WAIT, 'test'))
    assert guard.poll(observation) == 'stalled'


def test_new_focused_stage_releases_recenter_request():
    guard = NavigationGuard()
    guard.record(Observation(Scene.STAGES), Decision(Action.SCAN_STAGES, 'test'))
    assert guard.poll(Observation(Scene.STAGES, focused_stage=Stage(16, True, 0))) is None


def test_star_refinement_is_not_navigation_confirmation():
    guard = NavigationGuard()
    guard.record(Observation(Scene.RESULT, stage=1, stars=1), Decision(Action.DISMISS_RESULT, 'test'))
    assert guard.poll(Observation(Scene.RESULT, stage=1, stars=3)) == 'waiting'


def test_unknown_keeps_pending_request_but_new_scene_releases_it():
    guard = NavigationGuard()
    source = Observation(Scene.REWARD_INFO)
    guard.record(source, Decision(Action.CLOSE_INFO, 'test'))
    assert guard.poll(Observation(Scene.UNKNOWN)) is None
    assert guard.poll(source) == 'waiting'
    assert guard.poll(Observation(Scene.CHARACTERS)) is None
    assert guard.poll(source) is None


@pytest.mark.parametrize('action', [Action.BACK_TO_CHARACTERS, Action.SELECT_STAGE])
def test_map_control_or_focus_changes_do_not_prove_map_was_left(action):
    guard = NavigationGuard()
    source = Observation(Scene.STAGES, character_id='ttmr', focused_stage=Stage(16, True, 0))
    guard.record(source, Decision(action, 'test'))
    assert guard.poll(Observation(Scene.MAP_OVERVIEW, character_id='ttmr')) == 'waiting'
    assert guard.poll(Observation(Scene.STAGES, character_id='ttmr', focused_stage=Stage(17, False, 0))) == 'waiting'
    assert guard.poll(source) == 'stalled'


def test_overview_back_request_is_not_repeated_when_challenge_button_reappears():
    guard = NavigationGuard()
    guard.record(Observation(Scene.MAP_OVERVIEW, character_id='ttmr'), Decision(Action.BACK_TO_CHARACTERS, 'test'))
    assert guard.poll(Observation(Scene.STAGES, character_id='ttmr', focused_stage=Stage(16, True, 0))) == 'waiting'
    assert guard.poll(Observation(Scene.CHARACTERS)) is None
