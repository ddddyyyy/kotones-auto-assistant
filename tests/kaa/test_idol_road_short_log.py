"""短日志尾部的纯证据规则，不证明日志当前性。"""
from dataclasses import replace
import pytest
from kaa.idol_road.short_log import ShortLogEvidence,short_log_end_visible
from kaa.idol_road.battle_log import BattleLogBinding,LogViewport,LogEntryEvidence,tail_log_header,log_inspection_action
from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame
from kaa.idol_road.log_turns import LogTurnEvidence,confirm_visible_log_turn
from kaa.idol_road.rules import RuleTextRead,RuleTextLine

SESSION=BattleObservationSession('battle','ttmr',20,20,'confirmed_formation_start_request','variable_budget')


def proof(name='log'):
    return ShortLogEvidence(name,(138,148,542,48),(40,200,638,920),688,
        (40,700,638,420),(135,116,543,30),(684,128,3,1007),1.,1.,1.,(136,200,542,920),'c'*64)


def view(name='log',digest='b'*64):
    def reads(text):
        return tuple(RuleTextRead(name,kind,(RuleTextLine(text,.99),)) for kind in ('original','grayscale'))
    header=confirm_visible_log_turn(LogTurnEvidence(name,(164,158,98,31),.99,(580,162,88,22),.99,
        (138,148,129,48),reads('1ターン目'),(63,141,39,47),reads('20')))
    assert header.status=='resolved'
    return LogViewport(name,digest,(342,1140,36,38),(header,),None,(1.,1.),True,proof(name))


def hud(name):
    result=bind_battle_frame(SESSION,BattleFrameValues(name,True,20,50))
    assert result is not None
    return result


def entry(name):
    return LogEntryEvidence(name,(449,1179,30,39),.99,'a'*64)


def opened(clock=lambda:0):
    binding=BattleLogBinding(clock=clock)
    for name in ('one','two'):
        binding.observe_hud(hud(name),entry(name))
    binding.open(hud('two'),entry('two'))
    return binding


def test_no_scrollbar_alone_does_not_prove_tail():
    assert tail_log_header(replace(view(),short_list=None)) is None
    assert tail_log_header(view()).absolute_turn==1
    assert not tail_log_header(view()).currentness_confirmed


@pytest.mark.parametrize('changes',[
    {'frame_id':'stale'},{'top_clear_fraction':.998},{'track_clear_fraction':.998},
    {'tail_clear_fraction':.998},{'tail_rect':(40,1000,638,120)},
    {'tail_rect':(50,700,628,420)},{'last_content_y':705},
    {'track_rect':(684,128,3,500)},{'top_clear_rect':(135,148,543,30)},
    {'header_band':(138,400,542,48)},
])
def test_cropped_unknown_or_occluded_short_list_is_not_a_tail(changes):
    assert not short_log_end_visible(replace(proof(),**changes),'log',view().headers[0].evidence.suffix_rect)


def test_owned_short_log_requires_two_distinct_same_body_samples_before_closing():
    binding=opened()
    first=binding.inspect(view(),SESSION)
    assert first.tail_header is None and not first.short_list_stable
    assert log_inspection_action(first,'battle')=='wait'
    same=binding.inspect(view(),SESSION)
    assert same.tail_header is None
    second=binding.inspect(view('new'),SESSION)
    assert second.short_list_stable and second.tail_header.absolute_turn==1
    assert log_inspection_action(second,'battle')=='close'
    binding.close(second)
    returned=binding.observe_hud(hud('back'),entry('back'))
    assert returned.tail_header.absolute_turn==1 and returned.input_sequence_consistent
    assert not returned.currentness_confirmed and hud('back').observation.turn is None


def test_text_change_revokes_short_list_stability_and_does_not_renew_deadline():
    now=[0.]
    binding=opened(clock=lambda:now[0])
    binding.inspect(view(),SESSION)
    binding.inspect(view('new'),SESSION)
    changed_view=view('changed','c'*64)
    changed_view=replace(changed_view,short_list=replace(changed_view.short_list,text_fingerprint='d'*64))
    changed=binding.inspect(changed_view,SESSION)
    assert not changed.short_list_stable and changed.tail_header is None
    now[0]=30.
    expired=binding.inspect(view('still','d'*64),SESSION)
    assert expired.expired and log_inspection_action(expired,'battle')=='close'


def test_unbound_short_log_is_closed_without_adopting_or_waiting_for_currentness():
    inspected=BattleLogBinding().inspect(view(),None)
    assert inspected.request is None and inspected.tail_header is None
    assert log_inspection_action(inspected,'battle')=='close'


def test_input_change_revokes_even_a_stable_short_list_candidate():
    binding=opened()
    binding.inspect(view(),SESSION)
    assert binding.inspect(view('new'),SESSION).tail_header is not None
    binding.update_inputs(1,False)
    invalid=binding.inspect(view('after-input'),SESSION)
    assert invalid.expired and invalid.tail_header is None
    assert log_inspection_action(invalid,'battle')=='close'


def test_icon_animation_may_change_raw_pixels_but_not_text_and_header_stability():
    binding=opened()
    binding.inspect(view(),SESSION)
    animated=binding.inspect(view('icons-animated','d'*64),SESSION)
    assert animated.short_list_stable and animated.tail_header.absolute_turn==1
