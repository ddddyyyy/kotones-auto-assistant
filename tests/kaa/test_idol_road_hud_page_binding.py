from types import SimpleNamespace
import pytest
import numpy as np
from kaa.idol_road.page import IdolRoadPage
from kaa.idol_road.battle_log import BattleLogBinding
from kaa.idol_road.hud_initialization import InitialHudRound
from tests.kaa.test_idol_road_initialization import OWNER,setup
from tests.kaa.test_idol_road_hud_initialization import raw,hand


def prepared():
    inputs,_,_,ready=setup()
    page=IdolRoadPage.__new__(IdolRoadPage)
    page._battle_session=OWNER;page._battle_inputs=inputs;page._log_binding=BattleLogBinding(clock=lambda:0)
    page._maximum_hp=SimpleNamespace(session=OWNER,calibration=ready.maximum_hp_parameter.calibration,cancel_request=lambda:None)
    page.begin_hud_feedback(OWNER)
    producer=InitialHudRound(OWNER,deadline=90)
    producer.observe(inputs.snapshot(),raw('one',ready),hand('one'),now=1)
    current=raw('two',ready)
    producer.observe(inputs.snapshot(),current,hand('two'),now=2)
    page._hud_capture_frame=current
    page._hud_capture_hand=hand('two')
    return page,producer


def test_initial_hud_authorizes_the_exact_owned_frame_once_without_log_anchor():
    page,producer=prepared()
    bound=page.accept_initial_hud(producer)
    assert page.battle_rounds is None
    page.begin_battle_input(bound,'card','card')
    assert page.battle_inputs.sequence==1 and page.battle_inputs.pending.kind=='card'
    with pytest.raises(ValueError):page.begin_battle_input(bound,'skip')


def test_old_initial_frame_cannot_replace_the_latest_capture():
    page,producer=prepared();page._hud_capture_frame=None
    with pytest.raises(ValueError,match='最近采集'):page.accept_initial_hud(producer)


def test_an_intervening_unreadable_observation_revokes_input_from_the_old_frame():
    page,producer=prepared();bound=page.accept_initial_hud(producer)
    with pytest.raises(ValueError):page.observe(np.zeros((1,1,3),dtype='uint8'),'ttmr')
    with pytest.raises(ValueError,match='同帧回合授权'):page.begin_battle_input(bound,'skip')


def test_feedback_mode_cannot_be_restarted_after_an_input():
    page,producer=prepared();bound=page.accept_initial_hud(producer)
    page.begin_battle_input(bound,'skip')
    with pytest.raises(ValueError):page.begin_hud_feedback(OWNER)
