from dataclasses import replace
import pytest
from kaa.idol_road.recurring_hp_cost import RecurringHpTiming,recurring_hp_amount
from kaa.idol_road.rules import RuleTextRead,RuleTextLine


PARENT=(40,244,637,185)


def reads(amount='3'):
    return tuple(RuleTextRead('frame',kind,(RuleTextLine('以降、ターン開始時、',confidence),RuleTextLine('体力消費'+amount,.99)))
                 for kind,confidence in (('original',.95),('grayscale',.99)))


def timing():
    return RecurringHpTiming('frame',(160,322,503,95),(163,323,227,30),.999,True,(227,30))


@pytest.mark.parametrize('amount',['1','3','12'])
def test_full_phrase_and_current_strong_cost_pair_prove_amount_without_fixed_three(amount):
    assert recurring_hp_amount(reads(amount),PARENT,timing())==int(amount)


@pytest.mark.parametrize('changes',[
    {'frame_id':'old'},{'phrase_score':.969},{'unique':False},{'variant':'partial'},
    {'phrase_rect':(163,323,20,30)},{'phrase_rect':(100,323,227,30)},
    {'body_rect':(160,322,800,95)},{'template_size':(20,30)},
])
def test_invalid_incomplete_ambiguous_or_wrong_frame_template_cannot_rescue_weak_timing(changes):
    assert recurring_hp_amount(reads(),PARENT,replace(timing(),**changes)) is None


def test_template_never_overrides_original_strong_phase_or_numeric_conflict():
    base=reads()
    for line in (RuleTextLine('以降、ターン終了時、',.99),RuleTextLine('体力消費4',.99),RuleTextLine('体力回復3',.99)):
        lines=(line,base[1].lines[1]) if '時' in line.text else (base[1].lines[0],line)
        altered=(base[0],replace(base[1],lines=lines))
        assert recurring_hp_amount(altered,PARENT,timing()) is None


@pytest.mark.parametrize('altered',[
    (),reads()[:1],(reads()[0],reads()[0]),
    tuple(replace(read,frame_id='old') for read in reads()),
    tuple(replace(read,lines=(replace(read.lines[0],confidence=.95),read.lines[1])) for read in reads()),
    tuple(replace(read,lines=(read.lines[0],replace(read.lines[1],confidence=.97))) for read in reads()),
])
def test_missing_pair_stale_or_low_numeric_evidence_stays_unknown(altered):
    assert recurring_hp_amount(altered,PARENT,timing()) is None


def test_old_complete_strong_two_line_path_stays_available_without_new_template():
    strong=tuple(replace(read,lines=(replace(read.lines[0],confidence=.99),read.lines[1])) for read in reads())
    assert recurring_hp_amount(strong,PARENT,None)==3
    assert recurring_hp_amount(reads(),PARENT,None) is None


def test_both_weak_body_timings_need_strong_current_full_phrase_pair():
    weak=tuple(replace(r,lines=(replace(r.lines[0],confidence=.95),r.lines[1])) for r in reads())
    targeted=tuple(RuleTextRead('frame',kind,(RuleTextLine('以降、ターン開始時、',.99),))
        for kind in ('original','grayscale'))
    assert recurring_hp_amount(weak,PARENT,timing()) is None
    assert recurring_hp_amount(weak,PARENT,replace(timing(),phrase_reads=targeted))==3
    for bad in (targeted[:1],(targeted[0],targeted[0]),
        tuple(replace(r,frame_id='old') for r in targeted),
        tuple(replace(r,lines=(RuleTextLine('以降、ターン終了時、',.99),)) for r in targeted),
        tuple(replace(r,lines=(RuleTextLine('以降、ターン開始時、',.979),)) for r in targeted)):
        assert recurring_hp_amount(weak,PARENT,replace(timing(),phrase_reads=bad)) is None
