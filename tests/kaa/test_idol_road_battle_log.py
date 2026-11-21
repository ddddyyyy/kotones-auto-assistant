"""日志会话与有界请求测试；不模拟设备点击或宣称实机覆盖。"""
from dataclasses import replace
import pytest
from kaa.idol_road.battle_log import (BattleLogBinding,LogEntryEvidence,LogScrollEvidence,
    LogViewport,tail_log_header,log_at_bottom,log_inspection_action)
from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame
from kaa.idol_road.log_turns import LogTurnEvidence,confirm_visible_log_turn
from kaa.idol_road.rules import RuleTextRead,RuleTextLine
from kaa.idol_road.models import Scene,Observation,Action,Decision
from kaa.idol_road.page import PageSnapshot
from kaa.idol_road.planner_adapter import TrialFrameGuard
from kaa.idol_road.battle_trace import BattleTrace


SESSION=BattleObservationSession('battle','ttmr',20,20,'confirmed_formation_start_request','variable_budget')


def frame(name='hud',hp=48,remaining=2,session=SESSION):
    result=bind_battle_frame(session,BattleFrameValues(name,True,remaining,hp))
    assert result is not None
    return result


def entry(name='hud',digest='a'*64):
    return LogEntryEvidence(name,(449,1179,30,39),.99,digest)


def header(name='log',turn=18,remaining=2,y=436):
    def reads(text):
        return tuple(RuleTextRead(name,kind,(RuleTextLine(text,.99),)) for kind in ('original','grayscale'))
    evidence=LogTurnEvidence(name,(186,y,98,31),.99,(580,y+4,88,22),.99,
        (138,y-10,151,48),reads(f'{turn}ターン目'),(63,y-17,39,47),reads(str(remaining)))
    return confirm_visible_log_turn(evidence)


def view(name='log',bottom=True,digest='b'*64):
    scroll=LogScrollEvidence(name,(684,128,3,1007),(684,717 if bottom else 226,3,418),1.,589)
    return LogViewport(name,digest,(342,1140,36,38),(header(name),),scroll,(1.,1.),True)


def opened(clock=lambda:0):
    binding=BattleLogBinding(clock=clock)
    binding.observe_hud(frame('one'),entry('one'))
    binding.observe_hud(frame('two'),entry('two'))
    binding.open(frame('two'),entry('two'))
    return binding


def test_open_requires_two_distinct_stable_frames_and_exact_latest_proof():
    binding=BattleLogBinding()
    binding.observe_hud(frame(),entry())
    binding.observe_hud(frame(),entry())
    assert not binding.can_open(frame(),entry())
    binding.observe_hud(frame('new'),entry('new'))
    assert binding.can_open(frame('new'),entry('new'))
    assert not binding.can_open(frame(),entry())
    assert not binding.can_open(frame('new'),entry('new','c'*64))


@pytest.mark.parametrize('change',[
    {'hp':47},{'remaining':3},{'session':replace(SESSION,battle_id='other')},
    {'session':replace(SESSION,identity_source='historical_capture_metadata')}])
def test_changed_or_historical_hud_never_reuses_ready_identity(change):
    binding=BattleLogBinding()
    binding.observe_hud(frame('one'),entry('one'))
    binding.observe_hud(frame('two',**change),entry('two'))
    with pytest.raises(ValueError):
        binding.open(frame('two',**change),entry('two'))


def test_closed_owned_log_returns_diagnostic_only_and_never_changes_turn():
    binding=opened()
    inspected=binding.inspect(view(),SESSION)
    assert inspected.tail_header is not None
    binding.close(inspected)
    returned=binding.observe_hud(frame('back'),entry('back'))
    assert returned is not None and returned.hud_unchanged and returned.remaining_consistent
    assert returned.tail_header is not None and returned.tail_header.absolute_turn==18
    assert not returned.currentness_confirmed and frame('back').observation.turn is None
    assert not binding.pending


@pytest.mark.parametrize('change',[{'hp':47},{'remaining':1},{'digest':'c'*64}])
def test_return_changes_discard_tail_candidate(change):
    binding=opened()
    binding.close(binding.inspect(view(),SESSION))
    values={key:value for key,value in change.items() if key!='digest'}
    returned=binding.observe_hud(frame('back',**values),entry('back',change.get('digest','a'*64)))
    assert returned is not None and returned.tail_header is None


@pytest.mark.parametrize('change',[{'tail_clear_fractions':(1.,.9)},{'headers':()},
    {'body_fingerprint':'bad'},{'content_confirmed':False},{'scrollbar':None},
    {'headers':(header('wrong'),)}, {'headers':(header(),header(turn=17,y=800))},
    {'headers':(header(),replace(header(turn=19,y=800),status='partial',absolute_turn=None))}])
def test_incomplete_late_header_or_tail_cannot_be_replaced_by_maximum(change):
    assert tail_log_header(replace(view(),**change)) is None


def test_unbound_log_can_only_close_and_does_not_produce_return_evidence():
    binding=BattleLogBinding()
    inspection=binding.inspect(view(bottom=False),None)
    assert log_inspection_action(inspection,'battle')=='close'
    with pytest.raises(ValueError):
        binding.scroll(inspection)
    binding.close(inspection)
    with pytest.raises(ValueError):
        binding.close(inspection)
    assert binding.observe_hud(frame('back'),entry('back')) is None


def test_scroll_requires_thumb_and_body_progress_with_two_request_limit():
    binding=opened()
    first=binding.inspect(view(bottom=False),SESSION)
    assert log_inspection_action(first,'battle')=='scroll'
    binding.scroll(first)
    with pytest.raises(ValueError):
        binding.scroll(first)
    pending=binding.inspect(view('still',bottom=False),SESSION)
    assert pending.scroll_pending
    progressed=replace(view('progress',bottom=False,digest='c'*64),scrollbar=LogScrollEvidence(
        'progress',(684,128,3,1007),(684,300,3,418),1.,589))
    second=binding.inspect(progressed,SESSION)
    assert not second.scroll_pending
    binding.scroll(second)
    last=binding.inspect(replace(view('last',bottom=False,digest='d'*64),scrollbar=LogScrollEvidence(
        'last',(684,128,3,1007),(684,400,3,418),1.,589)),SESSION)
    assert last.scroll_attempts==2 and log_inspection_action(last,'battle')=='close'
    with pytest.raises(ValueError):
        binding.scroll(replace(last,scroll_attempts=0))


def test_no_scroll_progress_expires_and_allows_close_without_renewing_budget():
    binding=opened()
    binding.scroll(binding.inspect(view(bottom=False),SESSION))
    for name in ('still1','still2','still3'):
        inspection=binding.inspect(view(name,bottom=False),SESSION)
    assert inspection.expired and log_inspection_action(inspection,'battle')=='close'
    binding.close(inspection)


def test_repeated_frame_does_not_count_as_progress_and_timeout_is_absolute():
    now=[0.]
    binding=opened(clock=lambda:now[0])
    binding.scroll(binding.inspect(view(bottom=False),SESSION))
    for _ in range(10):
        same=binding.inspect(view(bottom=False),SESSION)
    assert not same.expired
    now[0]=30.
    expired=binding.inspect(view('new',bottom=False),SESSION)
    assert expired.expired and log_inspection_action(expired,'battle')=='close'


def test_session_change_drops_request_and_scroll_authority():
    binding=opened()
    inspection=binding.inspect(view(),replace(SESSION,battle_id='other'))
    assert inspection.request is None and inspection.tail_header is None


@pytest.mark.parametrize('change',[{'covered_fraction':.89},{'light_rows':0},
    {'thumb_rect':(684,128,3,1007)},{'thumb_rect':(684,717,4,418)}])
def test_unknown_scrollbar_does_not_mean_bottom(change):
    assert log_at_bottom(replace(view().scrollbar,**change)) is None


def test_trace_merges_identical_sampling_without_treating_frame_ids_as_progress():
    binding=opened()
    trace=BattleTrace()
    for name in ('log1','log2'):
        observation=Observation(Scene.BATTLE_LOG,'ttmr',20,battle_log_inspection=binding.inspect(view(name),SESSION))
        trace.observe(observation,Decision(Action.WAIT,'wait'))
    assert len(trace.snapshot().steps)==1
    assert trace.snapshot().steps[0].repeated_observations==2


def test_planner_waits_for_owned_log_and_rejects_unbound_or_stuck_return():
    now=[0.]
    guard=TrialFrameGuard(SESSION,100,clock=lambda:now[0])
    inspection=opened().inspect(view(),SESSION)
    assert guard.check(PageSnapshot(Observation(Scene.BATTLE_LOG,battle_log_inspection=inspection)))=='wait'
    now[0]=34
    assert guard.check(PageSnapshot(Observation(Scene.UNKNOWN)))=='wait'
    now[0]=35
    with pytest.raises(ValueError):
        guard.check(PageSnapshot(Observation(Scene.UNKNOWN)))
    with pytest.raises(ValueError):
        TrialFrameGuard(SESSION,100,clock=lambda:0).check(PageSnapshot(Observation(Scene.BATTLE_LOG,
            battle_log_inspection=BattleLogBinding().inspect(view(),None))))


def flow(mode='manual',dynamic=True):
    from kaa.config.idol_road import IdolRoadConfig
    from kaa.idol_road.controller import IdolRoadController
    from kaa.idol_road.models import Stage
    from kaa.idol_road.rules import StageRules,RuleViewport,VisibleStageRule
    from kaa.idol_road.rule_symbols import RuleSymbolEvidence
    controller=IdolRoadController(IdolRoadConfig(characters=['ttmr'],mode=mode))
    controller.step(Observation(Scene.STAGES,'ttmr',focused_stage=Stage(20,True,0)))
    controller.step(Observation(Scene.CONFIRM,'ttmr',20,stars=0,formation_ready=True,total_turns=20))
    if dynamic:
        title=tuple(RuleTextRead('frame',kind,(RuleTextLine('16ターン目',.99),)) for kind in ('original','grayscale'))
        body=tuple(replace(read,lines=(RuleTextLine('body unknown',.8),)) for read in title)
        row=VisibleStageRule((40,640,637,185),16,title,body,
            turn_addition=RuleSymbolEvidence('frame','turn_addition',(385,749,225,43),.99))
        controller._rules[('ttmr',20)]=StageRules('ttmr',20,(RuleViewport('page',(row,)),))
    return controller


def test_manual_variable_battle_inspects_once_and_closes_after_bottom():
    controller=flow()
    hud=Observation(Scene.BATTLE,'ttmr',20,battle_state=frame().observation,battle_log_available=True)
    assert controller.step(hud).action==Action.INSPECT_BATTLE_LOG
    binding=opened()
    panel=Observation(Scene.BATTLE_LOG,'ttmr',20,battle_log_inspection=binding.inspect(view(),SESSION))
    assert controller.step(panel).action==Action.CLOSE_BATTLE_LOG
    assert controller.step(hud).action==Action.WAIT
    assert controller.step(hud).reason=='manual_battle_in_progress'
    assert controller.battles_started==1


@pytest.mark.parametrize('dynamic,mode',[(False,'manual'),(True,'preview')])
def test_no_log_open_for_ordinary_or_preview_path(dynamic,mode):
    controller=flow(mode,dynamic)
    result=controller.step(Observation(Scene.BATTLE,'ttmr',20,
        battle_state=frame().observation,battle_log_available=True))
    assert result.action!=Action.INSPECT_BATTLE_LOG


def test_controller_unbound_log_is_never_scrolled():
    controller=flow()
    inspection=BattleLogBinding().inspect(view(bottom=False),None)
    assert controller.step(Observation(Scene.BATTLE_LOG,battle_log_inspection=inspection)).action==Action.CLOSE_BATTLE_LOG


@pytest.mark.parametrize('sequence,pending',[(1,False),(0,True)])
def test_input_change_while_log_is_open_keeps_close_authority_but_revokes_candidate(sequence,pending):
    binding=opened()
    request=binding.request
    binding.update_inputs(sequence,pending)
    inspection=binding.inspect(view(bottom=False),SESSION)
    assert inspection.request==request and inspection.inputs_invalidated and inspection.expired
    assert inspection.tail_header is None and log_inspection_action(inspection,'battle')=='close'
    with pytest.raises(ValueError):
        binding.scroll(replace(inspection,expired=False))
    binding.close(inspection)
    binding.update_inputs(sequence,False)
    returned=binding.observe_hud(frame('back'),entry('back'))
    assert returned is not None and returned.tail_header is None and not returned.input_sequence_consistent
    assert not returned.currentness_confirmed


def test_unchanged_input_sequence_is_a_separate_check_not_current_turn_authorization():
    binding=opened()
    binding.close(binding.inspect(view(),SESSION))
    returned=binding.observe_hud(frame('back'),entry('back'))
    assert returned is not None and returned.input_sequence_consistent
    assert returned.tail_header is not None and not returned.currentness_confirmed


def test_input_sequence_change_requires_two_new_stable_frames():
    binding=BattleLogBinding()
    for name in ('one','two'):
        binding.observe_hud(frame(name),entry(name))
    assert binding.can_open(frame('two'),entry('two'))
    binding.update_inputs(1,False)
    assert not binding.can_open(frame('two'),entry('two'))
    binding.observe_hud(frame('three'),entry('three'))
    assert not binding.can_open(frame('three'),entry('three'))
    binding.observe_hud(frame('four'),entry('four'))
    binding.open(frame('four'),entry('four'),input_sequence=1)
    assert binding.request is not None and binding.request.input_sequence==1


def test_pending_input_cannot_be_hidden_by_identical_hud():
    binding=BattleLogBinding()
    binding.update_inputs(1,True)
    for name in ('one','two','three'):
        binding.observe_hud(frame(name),entry(name))
    assert not binding.can_open(frame('three'),entry('three'))
    with pytest.raises(ValueError):
        binding.open(frame('three'),entry('three'),input_sequence=1)


def test_sequence_regression_or_round_trip_does_not_restore_old_candidate():
    binding=BattleLogBinding()
    binding.update_inputs(2,False)
    for name in ('one','two'):
        binding.observe_hud(frame(name),entry(name))
    binding.open(frame('two'),entry('two'),input_sequence=2)
    binding.update_inputs(3,False)
    binding.update_inputs(2,False)
    inspection=binding.inspect(view(),SESSION)
    assert inspection.inputs_invalidated and inspection.tail_header is None
    assert log_inspection_action(inspection,'battle')=='close'


@pytest.mark.parametrize('sequence,pending',[(True,False),(-1,False),(0,1)])
def test_invalid_input_metadata_is_rejected(sequence,pending):
    binding=BattleLogBinding()
    with pytest.raises(ValueError):
        binding.update_inputs(sequence,pending)


def test_closing_an_unbound_panel_does_not_clear_pending_input():
    binding=BattleLogBinding()
    binding.update_inputs(1,True)
    binding.close(binding.inspect(view(),None))
    for name in ('one','two','three'):
        binding.observe_hud(frame(name),entry(name))
    assert not binding.can_open(frame('three'),entry('three'))
