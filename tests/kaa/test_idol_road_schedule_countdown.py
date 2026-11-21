from dataclasses import replace
import pytest
from kaa.idol_road.rules import RuleTextLine,RuleTextRead
from kaa.idol_road.schedule_countdown import CountdownDigits,confirm_schedule_countdown,countdown_turn_candidates

BOX=(96,162,157,32)

def reads(text='7ターン後',confidence=.99,frame='frame'):
    return tuple(RuleTextRead(frame,kind,(RuleTextLine(text,confidence),)) for kind in ('original','grayscale'))

def digits(text='9',**changes):
    return replace(CountdownDigits('frame',(96,162,25,32),(121,170,82,24),.99,reads(text)),**changes)


def test_complete_current_countdown_requires_two_strong_readings():
    value=confirm_schedule_countdown('frame',BOX,reads())
    assert value.turns_after==7
    assert confirm_schedule_countdown('frame',BOX,reads(confidence=.97)).turns_after is None


def test_complete_suffix_and_whole_digit_prefix_can_corroborate_weak_full_text():
    value=confirm_schedule_countdown('frame',BOX,reads('9ターン後',.89),digits=digits())
    assert value.turns_after==9 and value.reads[0].lines[0].confidence==.89


@pytest.mark.parametrize('change',[{'suffix_score':.96},{'right_edge_ink':True},
    {'frame_id':'old'},{'rect':(97,162,24,32)},{'suffix_rect':(121,170,140,24)}])
def test_cropped_or_unconfirmed_prefix_never_recovers_a_countdown(change):
    assert confirm_schedule_countdown('frame',BOX,reads('9ターン後',.89),digits=digits(**change)).turns_after is None


@pytest.mark.parametrize('text',['7ターン後','ワターン後','9ターン前','09ターン後'])
def test_strong_conflicting_full_text_blocks_digit_fallback(text):
    assert confirm_schedule_countdown('frame',BOX,reads(text),digits=digits()).turns_after is None


def test_upscaled_reads_do_not_override_an_original_numeric_conflict():
    assert confirm_schedule_countdown('frame',BOX,reads('8ターン後'),reads('7ターン後')).turns_after is None


def test_visible_countdown_proposes_only_caller_supplied_possible_rounds():
    value=confirm_schedule_countdown('frame',BOX,reads('7ターン後'))
    assert countdown_turn_candidates(value,(10,16,18,20),(2,3))==(3,)
    assert countdown_turn_candidates(value,(10,11),(3,4))==(3,4)
    assert countdown_turn_candidates(value,(10,),(1,2))==()


def test_changed_value_cannot_be_accepted_without_its_original_reads():
    value=replace(confirm_schedule_countdown('frame',BOX,reads('7ターン後')),turns_after=8)
    assert countdown_turn_candidates(value,(10,),(2,3))==()


def test_scaled_whole_prefix_can_confirm_weak_original_without_lowering_threshold():
    proof=digits(reads=reads('9',.97),scaled_reads=reads('9'))
    value=confirm_schedule_countdown('frame',BOX,reads('9ターン後',.89),digits=proof)
    assert value.turns_after==9 and value.digits.reads==reads('9',.97)


@pytest.mark.parametrize('change',[
    {'reads':reads('8')}, {'reads':reads('9'),'scaled_reads':reads('8')},
    {'scaled_reads':reads('9',.97)}, {'scaled_reads':reads('9',frame='old')},
    {'scaled_reads':reads('09')}, {'right_edge_ink':True}])
def test_scaled_prefix_never_overrides_conflicts_weakness_or_clipping(change):
    proof=digits(reads=reads('9',.97),scaled_reads=reads('9'))
    proof=replace(proof,**change)
    assert confirm_schedule_countdown('frame',BOX,reads('9ターン後',.89),digits=proof).turns_after is None
