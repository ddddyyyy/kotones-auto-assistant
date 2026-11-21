"""卡牌只读采集的领域协议测试；没有设备操作或实战成功断言。"""
from dataclasses import replace
import pytest
from kaa.battle_turn import AbsoluteTurnObservation
from kaa.idol_road.battle_inputs import BattleInputs,BattleInputState,PendingBattleInput
from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame
from kaa.idol_road.battle_log import BattleLogBinding,LogEntryEvidence,LogViewport,LogScrollEvidence,LogRequest,LogReturn,log_inspection_action
from kaa.idol_road.card_probe import pending_card_permit,assess_card_log,card_feedback_ready
from kaa.idol_road.log_turns import LogTurnEvidence,confirm_visible_log_turn
from kaa.idol_road.rules import RuleTextRead,RuleTextLine
from kaa.idol_road.round_anchor import InitialRoundTracker
from kaa.tasks.produce.new.play_cards.planner.state import BattleState

SESSION=BattleObservationSession('synthetic-card','ttmr',19,20,'confirmed_formation_start_request')


def frame(name,remaining=20,hp=50):
    value=bind_battle_frame(SESSION,BattleFrameValues(name,True,remaining,hp))
    assert value is not None
    return value


def entry(name):
    return LogEntryEvidence(name,(449,1179,30,39),.99,'d'*64)


def view(turn,remaining):
    def reads(text):
        return tuple(RuleTextRead('log',kind,(RuleTextLine(text,.99),)) for kind in ('original','grayscale'))
    row=confirm_visible_log_turn(LogTurnEvidence('log',(168,436,98,31),.99,(580,440,88,22),.99,
        (138,426,133,48),reads(f'{turn}ターン目'),(63,419,39,47),reads(str(remaining))))
    return LogViewport('log','b'*64,(342,1140,36,38),(row,),
        LogScrollEvidence('log',(684,128,3,1007),(684,717,3,418),1.,589),(1.,1.),True)


def started():
    now=[0.]
    inputs=BattleInputs(SESSION,clock=lambda:now[0])
    inputs.begin('card',frame('before'),'card')
    inputs.bind_card_state(BattleState(20,50,0,hand_size=2,hand_card_ids=('card','other')),'card')
    permit=pending_card_permit(SESSION,inputs.snapshot(),now=0)
    assert permit is not None
    return inputs,permit,now


def opened(remaining=20):
    inputs,permit,now=started()
    binding=BattleLogBinding(clock=lambda:now[0])
    binding.update_inputs(1,True,card_probe=permit)
    for name in ('a','b'):
        binding.observe_hud(frame(name,remaining,48),entry(name))
    assert binding.can_open(frame('b',remaining,48),entry('b'))
    binding.open(frame('b',remaining,48),entry('b'),input_sequence=1,card_probe=permit)
    return inputs,permit,now,binding


def span_views():
    from kaa.idol_road.log_span import LogOverlap
    base=view(1,20)
    evidence=base.headers[0].evidence
    def moved(rect):return (rect[0],rect[1]-278,rect[2],rect[3])
    evidence=replace(evidence,frame_id='top',suffix_rect=moved(evidence.suffix_rect),hand_info_rect=moved(evidence.hand_info_rect),
        turn_rect=moved(evidence.turn_rect),remaining_rect=moved(evidence.remaining_rect),
        turn_reads=tuple(replace(r,frame_id='top') for r in evidence.turn_reads),
        remaining_reads=tuple(replace(r,frame_id='top') for r in evidence.remaining_reads))
    header=confirm_visible_log_turn(evidence)
    top=replace(base,frame_id='top',body_fingerprint='b'*64,headers=(header,),
        scrollbar=LogScrollEvidence('top',(684,128,3,1007),(684,128,3,901),1.,106),tail_clear_fractions=(1.,0.))
    bottom=replace(top,frame_id='bottom',body_fingerprint='c'*64,headers=(),content_confirmed=False,
        scrollbar=LogScrollEvidence('bottom',(684,128,3,1007),(684,234,3,901),1.,106),tail_clear_fractions=(1.,1.))
    proof=LogOverlap('top','bottom','b'*64,'c'*64,(136,200,542,920),121,.99,.5,1000)
    return top,bottom,proof


def test_owned_same_request_span_supplies_a_round_candidate_without_faking_a_visible_tail_or_settling_input():
    from kaa.idol_road.battle_log import tail_log_header
    inputs,permit,_,binding=opened()
    top,bottom,proof=span_views()
    first=binding.inspect(bottom,SESSION)
    assert log_inspection_action(first,SESSION.battle_id)=='scroll_header'
    original_request,original_started=binding.request,binding.started
    binding.scroll_to_header(first)
    second=binding.inspect(top,SESSION,overlap=proof)
    assert second.span is not None and second.tail_header is not None
    assert tail_log_header(top) is None and log_inspection_action(second,SESSION.battle_id)=='close'
    binding.close(second)
    # A closing-animation view must not replace the exact span evidence.
    binding.inspect(replace(top,frame_id='closing'),SESSION)
    assert binding.last_view==top and binding.started==original_started
    returned=binding.observe_hud(frame('back',hp=48),entry('back'))
    assert returned is not None and returned.request==original_request and returned.span==second.span
    result=assess_card_log(permit,returned,top,frame('back',hp=48),inputs.snapshot(),now=0)
    assert result.status=='same_round_observed' and result.round_evidence=='owned_viewport_span'
    assert not result.play_confirmed and inputs.pending is not None and permit.deadline==20


@pytest.mark.parametrize('kind',['request','sequence','source','digest','offset','score','alternative','no_signal','changed_height','new_header','header_not_offscreen'])
def test_span_rejects_stale_ambiguous_or_noncontiguous_evidence(kind):
    from kaa.idol_road.log_span import OwnedLogSpan,span_header
    _,_,_,binding=opened()
    top,bottom,proof=span_views()
    span=OwnedLogSpan(binding.request.request_id,SESSION.battle_id,1,top,bottom,proof)
    if kind=='request':span=replace(span,request_id='old')
    elif kind=='sequence':span=replace(span,input_sequence=2)
    elif kind=='source':span=replace(span,source='previous_log_title')
    elif kind=='digest':span=replace(span,overlap=replace(proof,top_body_digest='d'*64))
    elif kind=='offset':span=replace(span,overlap=replace(proof,offset=400))
    elif kind=='score':span=replace(span,overlap=replace(proof,score=.97))
    elif kind=='alternative':span=replace(span,overlap=replace(proof,alternative_score=.99))
    elif kind=='no_signal':span=replace(span,overlap=replace(proof,signal_pixels=0))
    elif kind=='changed_height':span=replace(span,bottom=replace(bottom,scrollbar=replace(bottom.scrollbar,thumb_rect=(684,234,3,890))))
    elif kind=='new_header':span=replace(span,bottom=replace(bottom,headers=top.headers))
    else:span=replace(span,overlap=replace(proof,offset=10))
    assert span_header(span,binding.request,top) is None


def test_header_recovery_is_bounded_and_does_not_renew_original_input_deadline():
    _,_,now,binding=opened()
    top,bottom,proof=span_views()
    binding.scroll_to_header(binding.inspect(bottom,SESSION))
    now[0]=20
    inspection=binding.inspect(top,SESSION,overlap=proof)
    assert inspection.expired and inspection.span is None
    assert log_inspection_action(inspection,SESSION.battle_id)=='close'
    assert binding.request.card_probe.deadline==20


def test_header_recovery_cannot_be_enabled_by_forging_an_inspection_flag():
    _,_,_,binding=opened()
    inspected=binding.inspect(view(1,20),SESSION)
    with pytest.raises(ValueError):binding.scroll_to_header(replace(inspected,header_scroll_needed=True))


def test_input_change_revokes_an_already_registered_span_before_hud_return():
    _,_,_,binding=opened()
    top,bottom,proof=span_views()
    binding.scroll_to_header(binding.inspect(bottom,SESSION))
    assert binding.inspect(top,SESSION,overlap=proof).span is not None
    binding.update_inputs(2,False)
    inspection=binding.inspect(top,SESSION,overlap=proof)
    assert inspection.expired and inspection.span is None and inspection.tail_header is None
    binding.close(inspection)
    returned=binding.observe_hud(frame('back',hp=48),entry('back'))
    assert returned is not None and returned.span is None and not returned.input_sequence_consistent


@pytest.mark.parametrize('at_bottom',[True,False])
def test_missing_long_log_header_is_diagnostic_only_and_never_reuses_an_earlier_title(at_bottom):
    inputs,permit,_,binding=opened()
    log=view(1,20)
    if not at_bottom:
        log=replace(log,scrollbar=replace(log.scrollbar,thumb_rect=(684,226,3,418)))
    log=replace(log,headers=(),content_confirmed=False)
    binding.close(binding.inspect(log,SESSION))
    returned=binding.observe_hud(frame('back',hp=48),entry('back'))
    resolution=assess_card_log(permit,returned,log,frame('back',hp=48),inputs.snapshot(),now=0)
    assert resolution.status=='unconfirmed' and resolution.turn is None
    assert resolution.unconfirmed_reason==('tail_header_unavailable' if at_bottom else None)
    assert inputs.pending is not None and not resolution.play_confirmed
    assert permit.deadline==20


def test_unrelated_log_frame_cannot_be_classified_as_current_missing_tail_header():
    inputs,permit,_,binding=opened()
    log=view(1,20)
    binding.close(binding.inspect(log,SESSION))
    returned=binding.observe_hud(frame('back',hp=48),entry('back'))
    different=replace(log,frame_id='other',headers=(),content_confirmed=False)
    resolution=assess_card_log(permit,returned,different,frame('back',hp=48),inputs.snapshot(),now=0)
    assert resolution.status=='unconfirmed' and resolution.unconfirmed_reason is None


def opened_hand_return():
    from kaa.idol_road.log_hand import LogHandCard,confirm_log_hand
    inputs,permit,now=started()
    binding=BattleLogBinding(clock=lambda:now[0])
    binding.update_inputs(1,True,card_probe=permit)
    def observation(name,upgrade=1):
        hand=confirm_log_hand(name,1,(LogHandCard('other',upgrade,True,(200,935,57,148),False),))
        assert hand is not None
        return replace(entry(name),hand_fingerprint=hand.fingerprint),hand
    for name in ('a','b'):
        proof,hand=observation(name)
        binding.observe_hud(frame(name,hp=48),proof,hand=hand)
    binding.open(frame('b',hp=48),proof,input_sequence=1,card_probe=permit)
    log=view(1,20)
    binding.close(binding.inspect(log,SESSION))
    return inputs,permit,now,binding,observation


def test_card_log_hand_mismatch_waits_for_one_fresh_complete_frame_without_renewing_request():
    inputs,permit,now,binding,observation=opened_hand_return()
    request,started_at=binding.request,binding.started
    original_inputs=inputs.snapshot()
    proof,hand=observation('sparkle',0)
    now[0]=16
    assert binding.observe_hud(frame('sparkle',hp=48),proof,hand=hand) is None
    assert binding.request==request and binding.started==started_at and binding.closing
    assert inputs.snapshot()==original_inputs
    # Mutating/replaying the same frame cannot turn the rejected frame into proof.
    proof,hand=observation('sparkle')
    assert binding.observe_hud(frame('sparkle',hp=48),proof,hand=hand) is None
    proof,hand=observation('fresh')
    now[0]=18
    returned=binding.observe_hud(frame('fresh',hp=48),proof,hand=hand)
    assert returned is not None and returned.hud_unchanged and returned.tail_header is not None
    assert returned.return_frame_id=='fresh' and returned.request==request
    assert returned.request.card_probe.deadline==permit.deadline==20
    assert not returned.currentness_confirmed and inputs.snapshot()==original_inputs


def test_card_log_second_mismatch_discards_candidate_and_does_not_keep_retrying():
    _,_,_,binding,observation=opened_hand_return()
    for name in ('first','second'):
        proof,hand=observation(name,0)
        result=binding.observe_hud(frame(name,hp=48),proof,hand=hand)
    assert result is not None and not result.hud_unchanged and result.tail_header is None
    assert not binding.pending


@pytest.mark.parametrize('change',[{'hp':47},{'remaining':19}])
def test_card_log_numeric_changes_are_not_hand_animation_rechecks(change):
    _,_,_,binding,observation=opened_hand_return()
    proof,hand=observation('changed',0)
    values={'hp':48,**change}
    result=binding.observe_hud(frame('changed',**values),proof,hand=hand)
    assert result is not None and result.tail_header is None and not binding.pending


def test_card_log_hand_recheck_does_not_survive_original_input_deadline():
    inputs,_,now,binding,observation=opened_hand_return()
    proof,hand=observation('sparkle',0)
    now[0]=19
    assert binding.observe_hud(frame('sparkle',hp=48),proof,hand=hand) is None
    now[0]=20
    proof,hand=observation('late')
    assert binding.observe_hud(frame('late',hp=48),proof,hand=hand) is None
    assert binding.expired and inputs.snapshot().expired


def test_card_log_hand_recheck_cannot_restore_invalidated_input_sequence():
    _,_,_,binding,observation=opened_hand_return()
    proof,hand=observation('sparkle',0)
    assert binding.observe_hud(frame('sparkle',hp=48),proof,hand=hand) is None
    binding.update_inputs(2,False)
    proof,hand=observation('changed')
    result=binding.observe_hud(frame('changed',hp=48),proof,hand=hand)
    assert result is not None and result.tail_header is None and not result.input_sequence_consistent


def test_partial_hand_recheck_cannot_authorize_a_round_but_a_fresh_complete_return_can():
    _,_,_,binding,observation=opened_hand_return()
    proof,hand=observation('partial',0)
    partial=frame('partial',hp=48)
    partial=replace(partial,values=replace(partial.values,detail_mode='core_only'))
    request=binding.request
    assert binding.observe_hud(partial,proof,hand=hand) is None and binding.request==request
    proof,hand=observation('complete')
    result=binding.observe_hud(frame('complete',hp=48),proof,hand=hand)
    assert result is not None and result.tail_header is not None and result.return_frame_id=='complete'


def test_partial_matching_hand_does_not_receive_a_log_round_candidate():
    _,_,_,binding,observation=opened_hand_return()
    proof,hand=observation('partial')
    partial=frame('partial',hp=48)
    partial=replace(partial,values=replace(partial.values,detail_mode='core_only'))
    result=binding.observe_hud(partial,proof,hand=hand)
    assert result is not None and result.hud_unchanged and result.tail_header is None


@pytest.mark.parametrize('turn,remaining,status',[(1,20,'same_round_observed'),(2,19,'next_round_observed'),
    (3,18,'header_gap')])
def test_read_only_card_return_classifies_round_without_settling_the_play(turn,remaining,status):
    inputs,permit,_,binding=opened(remaining)
    original=inputs.snapshot()
    log=view(turn,remaining)
    binding.close(binding.inspect(log,SESSION))
    returned=binding.observe_hud(frame('back',remaining,48),entry('back'))
    outcome=assess_card_log(permit,returned,log,frame('back',remaining,48),inputs.snapshot(),now=0)
    assert outcome.status==status and outcome.turn==turn and outcome.play_confirmed is False
    assert inputs.snapshot()==original
    assert card_feedback_ready(SESSION,inputs.snapshot(),frame('back',remaining,48),outcome,now=0)==(turn in (1,2))
    assert not card_feedback_ready(SESSION,inputs.snapshot(),frame('later',remaining,48),outcome,now=0)
    assert not card_feedback_ready(SESSION,inputs.snapshot(),frame('back',remaining,47),outcome,now=0)
    assert not card_feedback_ready(SESSION,inputs.snapshot(),frame('back',remaining,48),outcome,now=20)
    assert binding.request is None
    for kind,card_id in (('card','other'),('skip',None)):
        with pytest.raises(ValueError):
            inputs.begin(kind,frame('later',remaining,48),card_id)


@pytest.mark.parametrize('kind',['unbound','wrong_card','hp','remaining','projection','absolute','sequence','expired','historical'])
def test_card_permit_requires_bound_observed_before_and_original_input(kind):
    inputs,_,_=started()
    snapshot=inputs.snapshot()
    pending=snapshot.pending
    before=pending.before
    if kind=='unbound':
        pending=replace(pending,before=None)
    elif kind=='wrong_card':
        pending=replace(pending,kind='skip',card_id=None)
    elif kind=='hp':
        pending=replace(pending,before=replace(before,hp=49))
    elif kind=='remaining':
        pending=replace(pending,before=replace(before,remaining_turns=19))
    elif kind=='projection':
        pending=replace(pending,before=replace(before,projected_effect_values=()))
    elif kind=='absolute':
        pending=replace(pending,before=replace(before,absolute_turn_observation=AbsoluteTurnObservation('other','before',1,20)))
    elif kind=='sequence':
        snapshot=replace(snapshot,sequence=2)
    elif kind=='expired':
        snapshot=replace(snapshot,expired=True)
    else:
        pending=replace(pending,frame=replace(pending.frame,session=replace(SESSION,identity_source='historical_capture_metadata')))
    assert pending_card_permit(SESSION,replace(snapshot,pending=pending),now=0) is None


@pytest.mark.parametrize('field,value',[('hud_unchanged',False),('remaining_consistent',False),
    ('input_sequence_consistent',False),('return_frame_id','old'),('tail_header',None)])
def test_incomplete_return_cannot_supply_card_feedback_round(field,value):
    inputs,permit,_,binding=opened()
    log=view(1,20)
    binding.close(binding.inspect(log,SESSION))
    returned=binding.observe_hud(frame('back',20,48),entry('back'))
    outcome=assess_card_log(permit,replace(returned,**{field:value}),log,frame('back',20,48),inputs.snapshot(),now=0)
    assert outcome.status=='unconfirmed' and inputs.pending is not None


def test_card_read_deadline_does_not_restart_when_log_opens_or_closes():
    inputs,permit,now,binding=opened()
    assert permit.deadline==inputs.pending.started+20
    log=view(1,20)
    binding.inspect(log,SESSION)
    now[0]=20
    inspected=binding.inspect(log,SESSION)
    assert inspected.expired and inspected.tail_header is None
    assert log_inspection_action(inspected,SESSION.battle_id)=='close'
    binding.close(inspected)
    assert pending_card_permit(SESSION,inputs.snapshot(),now=now[0]) is None
    assert not binding.can_open(frame('late'),entry('late'))


def test_bound_before_is_required_even_to_accumulate_readiness():
    inputs,permit,_=started()
    bad=replace(permit,pending=replace(permit.pending,before=None))
    binding=BattleLogBinding(clock=lambda:0)
    binding.update_inputs(1,True,card_probe=bad)
    for name in ('a','b'):
        binding.observe_hud(frame(name),entry(name))
    assert not binding.can_open(frame('b'),entry('b'))
    with pytest.raises(ValueError):
        binding.open(frame('b'),entry('b'),input_sequence=1,card_probe=bad)
    assert inputs.pending is not None


def test_rebinding_before_invalidates_existing_candidate_even_at_same_sequence():
    inputs,permit,_,binding=opened()
    changed=replace(permit,pending=replace(permit.pending,before=replace(permit.pending.before,genki=3)))
    binding.update_inputs(1,True,card_probe=changed)
    binding.update_inputs(1,True,card_probe=permit)
    inspected=binding.inspect(view(1,20),SESSION)
    assert inspected.inputs_invalidated and inspected.tail_header is None
    assert inputs.pending is not None


def test_old_title_after_late_input_does_not_move_round_backwards():
    inputs=BattleInputs(SESSION,clock=lambda:0)
    inputs.begin('card',frame('before',19),'card')
    inputs.bind_card_state(BattleState(19,50,0),'card')
    permit=pending_card_permit(SESSION,inputs.snapshot(),now=0)
    binding=BattleLogBinding(clock=lambda:0)
    binding.update_inputs(1,True,card_probe=permit)
    for name in ('a','b'):
        binding.observe_hud(frame(name,19),entry(name))
    binding.open(frame('b',19),entry('b'),input_sequence=1,card_probe=permit)
    # fixed预算帧本身已表明turn2，与旧title1冲突，不应提供任何候选。
    log=view(1,19)
    binding.close(binding.inspect(log,SESSION))
    returned=binding.observe_hud(frame('back',19),entry('back'))
    assert assess_card_log(permit,returned,log,frame('back',19),inputs.snapshot(),now=0).status=='unconfirmed'


@pytest.mark.parametrize('turn,remaining,status',[(1,20,'same_round_observed'),(2,19,'next_round_observed'),
    (2,20,'next_round_observed'),(2,21,'next_round_observed'),(3,20,'header_gap')])
def test_variable_card_protocol_uses_title_not_remaining_delta_without_inventing_a_round(turn,remaining,status):
    session=replace(SESSION,turn_budget_mode='variable_budget')
    def raw(name,count=20,hp=50):
        return bind_battle_frame(session,BattleFrameValues(name,True,count,hp))
    tracker=InitialRoundTracker(session)
    initial=view(1,20)
    returned=LogReturn(LogRequest('initial',session,'open',20,50,'a'*64),'log','seed',True,True,
                       initial.headers[0],input_sequence_consistent=True)
    empty=BattleInputState(session.battle_id,0,None,False)
    assert tracker.accept(raw('seed'),returned,initial,empty)
    origin=tracker.bind(raw('seed'),empty)
    # 合成领域输入不访问设备；无锚点卡牌仍拒绝。
    ledger=BattleInputs(session,clock=lambda:0)
    with pytest.raises(ValueError):
        ledger.begin('card',raw('unanchored'),'card')
    ledger.begin('card',origin,'card')
    before=BattleState(20,50,0,absolute_turn_observation=origin.absolute_turn)
    ledger.bind_card_state(before,'card')
    inputs=ledger.snapshot()
    pending=inputs.pending
    permit=pending_card_permit(session,inputs,now=0)
    assert permit is not None
    wrong=replace(pending,before=replace(before,absolute_turn_observation=replace(origin.absolute_turn,frame_id='other')))
    assert pending_card_permit(session,replace(inputs,pending=wrong),now=0) is None
    binding=BattleLogBinding(clock=lambda:0)
    binding.update_inputs(1,True,card_probe=permit)
    for name in ('a','b'):
        binding.observe_hud(raw(name,remaining,48),entry(name))
    binding.open(raw('b',remaining,48),entry('b'),input_sequence=1,card_probe=permit)
    log=view(turn,remaining)
    binding.close(binding.inspect(log,session))
    returned=binding.observe_hud(raw('back',remaining,48),entry('back'))
    outcome=assess_card_log(permit,returned,log,raw('back',remaining,48),inputs,now=0)
    assert outcome.status==status and not outcome.play_confirmed
    assert not card_feedback_ready(session,inputs,raw('back',remaining,48),outcome,now=0)
    assert raw('back',remaining,48).absolute_turn is None and inputs.pending==pending


@pytest.mark.parametrize('turn,remaining,changed',[(1,20,True),(1,20,False),(2,19,False)])
def test_log_return_defers_settlement_to_existing_card_feedback(turn,remaining,changed):
    from kaa.idol_road.input_policy import InputBoundPolicy
    from kaa.tasks.produce.new.play_cards.planner.feedback import PendingPlay,evaluate_play_feedback,PlayFeedbackStatus
    inputs,permit,_,binding=opened(remaining)
    log=view(turn,remaining)
    binding.close(binding.inspect(log,SESSION))
    after_frame=frame('back',remaining,48)
    returned=binding.observe_hud(after_frame,entry('back'))
    outcome=assess_card_log(permit,returned,log,after_frame,inputs.snapshot(),now=0)
    assert card_feedback_ready(SESSION,inputs.snapshot(),after_frame,outcome,now=0)
    assert inputs.pending is not None and outcome.play_confirmed is False
    before=permit.pending.before
    pending=PendingPlay('card',before,0,0)
    # 未观察到可靠资源时，不把日志所示回合当成卡牌成功。
    after=replace(before,remaining_turns=remaining,hp=48,hp_known=changed,
                  hand_size=1 if changed else 2,hand_card_ids=('other',) if changed else ('card','other'))
    feedback=evaluate_play_feedback(pending,after)
    policy=InputBoundPolicy(inputs,lambda:after_frame)
    if changed:
        assert feedback.status is PlayFeedbackStatus.CONFIRMED and feedback.evidence=='card_left_hand'
        policy.feedback(pending,feedback,after)
        assert inputs.pending is None
    elif turn==2:
        assert feedback.status is PlayFeedbackStatus.UNCONFIRMED
        policy.feedback(pending,feedback,after)
        assert inputs.pending is None  # 结束旧等待，未追认出牌成功。
    else:
        assert feedback.status is PlayFeedbackStatus.UNCONFIRMED
        with pytest.raises(ValueError):
            policy.feedback(pending,feedback,after)
        assert inputs.pending is not None


def test_page_syncs_card_permit_without_falsifying_pending_as_idle():
    from kaa.idol_road.page import IdolRoadPage
    inputs,permit,_=started()
    page=object.__new__(IdolRoadPage)
    page._battle_session=SESSION
    page._battle_inputs=inputs
    page._log_binding=BattleLogBinding(clock=lambda:0)
    assert page._sync_log_inputs()==permit
    assert page._log_binding._input_pending is True
    assert page._log_binding._card_probe==permit
    assert inputs.pending==permit.pending
