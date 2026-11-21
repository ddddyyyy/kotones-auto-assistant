from dataclasses import replace
import pytest
from kaa.idol_road.rules import RuleTextRead,RuleTextLine
from kaa.idol_road.skip_recovery import SkipRecoveryEvidence,confirm_skip_recovery


def evidence(text='2'):
    reads=tuple(RuleTextRead('frame',kind,(RuleTextLine(text,.999),)) for kind in ('original','grayscale'))
    return SkipRecoveryEvidence('frame',True,(644,750,44,38),(652,754,26,30),(654,756,23,26),.99,False,reads)


def test_explicit_single_digit_recovery_is_display_evidence_without_future_hp_guarantee():
    for text in ('0','2','9'):
        result=confirm_skip_recovery(evidence(text))
        assert result.status=='resolved' and result.value==int(text)
        assert result.scope=='current_button_recovery_display_only' and not result.next_turn_hp_guaranteed


@pytest.mark.parametrize('changes',[{'enabled':False},{'edge_clipped':True},{'edge_clipped':None},
    {'icon_score':.96},{'icon_score':float('nan')},{'icon_score':True},{'icon_score':None},
    {'number_rect':(654,756,25,26)},{'number_mask_rect':(640,754,26,30)},
    {'reads':()},{'frame_id':'other'}])
def test_missing_disabled_clipped_cross_frame_or_bad_icon_evidence_is_not_zero(changes):
    result=confirm_skip_recovery(replace(evidence(),**changes))
    assert result.status=='unknown' and result.value is None


@pytest.mark.parametrize('text',['02','+2','2HP','2 3','', '12'])
def test_only_the_audited_complete_single_digit_layout_is_supported(text):
    assert confirm_skip_recovery(evidence(text)).value is None


def test_numeric_conflict_and_low_confidence_are_not_repaired():
    original=evidence()
    for changed in (replace(original.reads[1],lines=(RuleTextLine('3',.999),)),
                    replace(original.reads[1],lines=(RuleTextLine('2',.97),))):
        assert confirm_skip_recovery(replace(original,reads=(original.reads[0],changed))).value is None
