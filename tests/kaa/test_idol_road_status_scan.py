from dataclasses import replace
import pytest
from kaa.config.idol_road import IdolRoadConfig
from kaa.idol_road.controller import IdolRoadController
from kaa.idol_road.models import Action,Observation,Scene
from kaa.idol_road.status_details import BattleStatusDetails,StatusRowEvidence,confirm_status_row
from kaa.idol_road.status_scan import status_viewport_key
from kaa.idol_road.rules import RuleTextRead,RuleTextLine


def details(frame='frame',y=620,occluded=True,**changes):
    def reads(text):
        return tuple(RuleTextRead(frame,kind,(RuleTextLine(text,.99),)) for kind in ('original','grayscale'))
    evidence=StatusRowEvidence(frame,(145,y,56,29),.99,reads('好調'),reads('1ターン'),
        reads('スコア上昇量を50%増加1ターン'),occluded=occluded)
    return replace(BattleStatusDetails(frame,'battle','ttmr',16,(confirm_status_row(evidence),),
        identity_source='confirmed_formation_start_request'),**changes)


def active_flow():
    flow=IdolRoadController(IdolRoadConfig(characters=['ttmr'],mode='manual'))
    # 本测试只覆盖已请求的详情采集策略；开战／打开门控另有独立测试。
    flow._stage=16
    flow._pending_status=('battle',3)
    return flow


def observation(value):
    return Observation(Scene.BATTLE_STATUS_DETAILS,'ttmr',16,battle_status_details=value,
        battle_status_scroll_available=True)


def test_frame_ids_and_ocr_confidence_do_not_prove_a_scroll():
    before=details()
    fresh=details('fresh')
    assert status_viewport_key(before)==status_viewport_key(fresh)
    assert status_viewport_key(before)!=status_viewport_key(details('moved',y=573,occluded=False))


def test_clipped_detail_scrolls_once_then_closes_after_position_changes():
    flow=active_flow()
    assert flow.step(observation(details())).action==Action.SCROLL_BATTLE_STATUS
    assert flow.step(observation(details('stale'))).action==Action.WAIT
    assert flow.step(observation(details('moved',y=573,occluded=False))).action==Action.CLOSE_BATTLE_STATUS
    assert flow._pending_status is None and flow._status_scans[('battle',3)].scrolls==1
    assert len(flow._status_scans[('battle',3)].viewports)==2


def test_stale_scroll_has_bounded_wait_and_never_another_swipe():
    flow=active_flow()
    assert flow.step(observation(details())).action==Action.SCROLL_BATTLE_STATUS
    assert flow.step(observation(details('b'))).action==Action.WAIT
    end=flow.step(observation(details('c')))
    assert end.action==Action.CLOSE_BATTLE_STATUS and end.reason=='battle_status_scroll_not_confirmed'


@pytest.mark.parametrize('change',[{'battle_id':'other'},{'stage':15},{'character_id':'hski'},
    {'identity_source':'historical_capture_metadata'},{'frame_id':''}])
def test_unbound_conflicting_or_historical_details_only_close(change):
    flow=active_flow()
    assert flow.step(observation(details(**change))).action==Action.CLOSE_BATTLE_STATUS
    assert not flow._status_scans


def test_manual_panel_and_unclipped_panel_are_not_scrolled():
    flow=IdolRoadController(IdolRoadConfig(characters=['ttmr'],mode='manual'))
    assert flow.step(observation(details())).action==Action.CLOSE_BATTLE_STATUS
    flow=active_flow()
    assert flow.step(observation(details(y=437,occluded=False))).action==Action.CLOSE_BATTLE_STATUS


def test_inconsistent_or_missing_row_evidence_does_not_generate_viewport_key():
    value=details()
    assert status_viewport_key(replace(value,rows=())) is None
    assert status_viewport_key(replace(value,rows=(replace(value.rows[0],remaining_turns=999),))) is None
    assert status_viewport_key(replace(value,frame_id='old')) is None


def test_cancel_precedes_status_scroll():
    flow=active_flow()
    assert flow.step(observation(details()),cancelled=True).reason=='cancelled'
