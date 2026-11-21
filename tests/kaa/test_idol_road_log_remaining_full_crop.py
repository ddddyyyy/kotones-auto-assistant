from dataclasses import replace
import pytest
from kaa.idol_road.log_turns import LogRemainingDirect,confirm_visible_log_turn
from kaa.idol_road.log_remaining_glyphs import LogRemainingGlyphs
from kaa.idol_road.rules import RuleTextRead,RuleTextLine
from tests.kaa.test_idol_road_log_turns import evidence,reads


def bounded_evidence():
    base=evidence()
    raw=(RuleTextRead('frame','original',(RuleTextLine('18]',.95),)),
         RuleTextRead('frame','grayscale',(RuleTextLine('18]',.94),RuleTextLine('ATA',.54))))
    scaled=(reads('18')[0],RuleTextRead('frame','grayscale',(
        RuleTextLine('18',.99),RuleTextLine('4T',.56))))
    glyphs=LogRemainingGlyphs('frame',base.remaining_rect,
        ((21,7,14,22,163),(3,8,13,20,89)),((21,7,14,22,163),(3,8,13,20,90)),252,253)
    return replace(base,remaining_reads=raw,remaining_scaled_reads=scaled,
        remaining_glyphs=glyphs,remaining_direct=LogRemainingDirect('frame',base.remaining_rect,reads('18'),False),
        remaining_direct_scaled=LogRemainingDirect('frame',base.remaining_rect,reads('18'),False,scale=2))


def test_four_whole_crop_readings_and_complete_digit_masks_resolve_detector_noise_without_string_repair():
    current=bounded_evidence()
    assert confirm_visible_log_turn(replace(current,remaining_direct_scaled=None)).remaining_turns is None
    result=confirm_visible_log_turn(current)
    assert (result.absolute_turn,result.remaining_turns,result.status)==(17,18,'resolved')
    assert result.evidence.remaining_reads[0].lines[0].text=='18]'


@pytest.mark.parametrize('change',['stale','wrong_number','low_confidence','wrong_scale','wrong_rect','edge'])
def test_incomplete_or_conflicting_recognition_does_not_rescue_remaining_number(change):
    current=bounded_evidence()
    scaled=current.remaining_direct_scaled
    changed={'stale':{'frame_id':'old'},'wrong_number':{'reads':reads('19')},
        'low_confidence':{'reads':reads('18',.979)},'wrong_scale':{'scale':1},
        'wrong_rect':{'rect':(64,419,39,47)},'edge':{'right_edge_clipped':True}}[change]
    assert confirm_visible_log_turn(replace(current,remaining_direct_scaled=replace(scaled,**changed))).remaining_turns is None


@pytest.mark.parametrize('change',['extra_glyph','cropped','different_masks','missing_pixels','stale'])
def test_whole_glyph_geometry_rejects_extra_content_and_mask_mismatch(change):
    current=bounded_evidence()
    glyphs=current.remaining_glyphs
    changed={
        'extra_glyph':{'neutral_components':(*glyphs.neutral_components,(1,35,3,5,15))},
        'cropped':{'neutral_components':((21,7,18,22,163),(3,8,13,20,89))},
        'different_masks':{'grayscale_components':((20,7,14,22,163),(3,8,13,20,90))},
        'missing_pixels':{'neutral_pixels':251},'stale':{'frame_id':'old'},
    }[change]
    assert confirm_visible_log_turn(replace(current,remaining_glyphs=replace(glyphs,**changed))).remaining_turns is None


@pytest.mark.parametrize('text',['19','18]','18A'])
def test_strong_detector_conflict_cannot_be_ignored(text):
    current=bounded_evidence()
    assert confirm_visible_log_turn(replace(current,remaining_reads=reads(text))).remaining_turns is None


def test_resolving_remaining_does_not_override_absolute_turn_conflict():
    current=bounded_evidence()
    result=confirm_visible_log_turn(replace(current,turn_reads=(reads('17ターン目')[0],reads('18ターン目')[1])))
    assert result.remaining_turns==18 and result.absolute_turn is None and result.status=='conflict'
