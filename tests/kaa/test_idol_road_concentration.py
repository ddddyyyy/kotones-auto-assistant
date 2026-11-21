from dataclasses import replace
import pytest
from kaa.idol_road.concentration import ConcentrationNumberRead, confirm_concentration
from kaa.idol_road.battle_observation import BattleFrameValues,BattleObservationSession,bind_battle_frame


def reads(text='10'):
    return tuple(ConcentrationNumberRead('frame',kind,text,.99,(58,266,56,24))
                 for kind in ('original','color_white_mask','gray_white_mask'))


def observation(numbers=None):
    return confirm_concentration('frame',(28,247,31,28),.99,numbers if numbers is not None else reads())


def test_status_inspection_target_requires_current_recomputed_evidence():
    from kaa.idol_road.battle_observation import status_inspection_target
    session=BattleObservationSession('battle','ttmr',16,10,'confirmed_formation_start_request')
    values=BattleFrameValues('frame',True,8,40,concentration=observation())
    frame=bind_battle_frame(session,values)
    assert status_inspection_target(frame)==(28,247,31,28)
    assert status_inspection_target(replace(frame,session=replace(session,battle_id='other'))) is None
    for changed in (replace(values,resource_state='selection_preview'),
                    replace(values,concentration=None),replace(values,frame_id='next'),
                    replace(values,concentration=replace(observation(),value=99))):
        assert status_inspection_target(bind_battle_frame(session,changed)) is None
    historical=replace(session,identity_source='historical_capture_metadata')
    assert status_inspection_target(bind_battle_frame(historical,values)) is None


def test_current_known_icon_can_open_details_without_inventing_its_unknown_number():
    from kaa.idol_road.battle_observation import status_inspection_target
    session=BattleObservationSession('battle','ttmr',16,10,'confirmed_formation_start_request')
    primary,color,gray=reads('3')
    icon=observation((primary,replace(color,text=None,confidence=None,row_count=0),gray))
    assert icon.status=='number_unknown'
    frame=bind_battle_frame(session,BattleFrameValues('frame',True,8,48,concentration=icon))
    assert frame.observation.concentration is None
    assert status_inspection_target(frame)==(28,247,31,28)
    stale=observation((replace(primary,frame_id='old'),color,gray))
    assert status_inspection_target(bind_battle_frame(session,BattleFrameValues('frame',True,8,48,concentration=stale))) is None


@pytest.mark.parametrize('text,value',[('0',0),('4',4),('7',7),('10',10),('100',100)])
def test_full_same_frame_numbers_confirmed_by_both_masks(text,value):
    result=observation(reads(text))
    assert result.status=='resolved' and result.value==value


@pytest.mark.parametrize('text',['+4','7x','01','10/20','-1'])
def test_high_confidence_non_full_numeric_text_is_not_trimmed(text):
    assert observation(reads(text)).status=='conflict'


def test_missing_icon_never_means_zero_concentration():
    assert confirm_concentration('frame',None,None,reads('0')).value is None
    assert confirm_concentration('frame',(28,247,31,28),.96,reads()).status=='icon_unknown'


def test_same_crop_double_size_rechecks_complete_agreeing_proposal():
    primary,color,gray=reads('9')
    base=(primary,replace(color,confidence=.939),gray)
    scaled=tuple(replace(row,scale=2) for row in reads('9'))
    result=confirm_concentration('frame',(28,369,31,28),.999,base,scaled)
    assert result.value==9 and result.value_source=='same_crop_double_size'
    assert result.reads==base and result.scaled_reads==scaled
    bound=bind_battle_frame(session(),BattleFrameValues('frame',True,8,40,concentration=result))
    assert bound is not None and bound.observation.concentration==9


@pytest.mark.parametrize('kind',['conflict','stale','different_crop','wrong_scale','clipped','incomplete','disagree'])
def test_enlargement_cannot_override_conflicts_or_missing_original_evidence(kind):
    primary,color,gray=reads('9')
    base=(primary,replace(color,confidence=.939),gray)
    scaled=tuple(replace(row,scale=2) for row in reads('9'))
    if kind=='conflict':
        base=(replace(primary,text='19'),base[1],gray)
    elif kind=='stale':
        scaled=(replace(scaled[0],frame_id='old'),*scaled[1:])
    elif kind=='different_crop':
        scaled=tuple(replace(row,rect=(58,267,56,24)) for row in scaled)
    elif kind=='wrong_scale':
        scaled=tuple(replace(row,scale=1) for row in scaled)
    elif kind=='clipped':
        base=(primary,replace(base[1],right_edge_ink=True),gray)
    elif kind=='incomplete':
        base=(replace(primary,text=None,confidence=None,row_count=0),*base[1:])
    else:
        base=(replace(primary,text='8',confidence=.97),replace(color,text='9',confidence=.97),replace(gray,confidence=.97))
    result=confirm_concentration('frame',(28,369,31,28),.999,base,scaled)
    assert result.value is None


def test_high_confidence_original_conflict_is_not_overridden_by_masks():
    primary,color,gray=reads('7')
    assert observation((replace(primary,text='4'),color,gray)).status=='conflict'
    assert observation((primary,color,replace(gray,text='4'))).status=='conflict'


def test_low_confidence_original_can_be_verified_by_same_whole_crop_masks():
    primary,color,gray=reads('7')
    assert observation((replace(primary,confidence=.96),color,gray)).value==7
    assert observation((primary,replace(color,confidence=.97),gray)).value is None


def test_cross_frame_different_crop_missing_or_duplicate_representations_remain_unknown():
    primary,color,gray=reads()
    for invalid in ((primary,color), (primary,color,color),
                    (replace(primary,frame_id='old'),color,gray),
                    (primary,replace(color,rect=(59,266,56,24)),gray)):
        assert observation(invalid).status=='number_unknown'


def test_multiple_raw_ocr_rows_cannot_be_silently_discarded():
    primary,color,gray=reads()
    assert observation((replace(primary,row_count=2,text=None,confidence=None),color,gray)).status=='conflict'


def test_possible_right_edge_digit_clipping_does_not_accept_numeric_prefix():
    primary,color,gray=reads('100')
    assert observation((primary,replace(color,right_edge_ink=True),gray)).status=='number_unknown'


def session():
    return BattleObservationSession('battle','ttmr',16,10,'confirmed_formation_start_request')


def test_verified_value_flows_to_rule_state_but_conflicts_or_stale_frames_do_not():
    frame=bind_battle_frame(session(),BattleFrameValues('frame',True,8,40,concentration=observation()))
    assert frame is not None and frame.observation.concentration==10 and frame.observation.turn==3
    stale=bind_battle_frame(session(),BattleFrameValues('other-frame',True,8,40,concentration=observation()))
    assert stale is not None and stale.observation.concentration is None and 'concentration' in stale.observation.conflicted_fields
    forged=bind_battle_frame(session(),BattleFrameValues('frame',True,8,40,concentration=replace(observation(),value=20)))
    assert forged is not None and forged.observation.concentration is None


@pytest.mark.parametrize('state',['selection_preview','unconfirmed_overlay'])
def test_preview_values_never_become_confirmed_hp_turn_or_concentration(state):
    frame=bind_battle_frame(session(),BattleFrameValues('frame',True,8,40,resource_state=state,concentration=observation()))
    assert frame is not None and frame.observation.source=='unconfirmed_resource_preview'
    assert frame.observation.concentration is None and frame.observation.hp is None and frame.observation.turn is None


def neutral_reads(text='4'):
    return tuple(ConcentrationNumberRead('frame',kind,text,.999,(58,266,56,24),scale=2)
        for kind in ('neutral_white_mask','neutral_gray_white_mask'))


def interrupted_masks():
    original,color,gray=reads('4')
    return original,replace(color,text='4 1',confidence=.77),gray


def test_neutral_white_recheck_requires_original_and_another_high_confidence_proposal():
    result=confirm_concentration('frame',(28,247,31,28),.99,interrupted_masks(),neutral_reads=neutral_reads())
    assert result.value==4 and result.value_source=='same_crop_neutral_white'
    values=BattleFrameValues('frame',True,8,40,concentration=result)
    bound=bind_battle_frame(session(),values)
    assert bound.observation.concentration==4
    from kaa.idol_road.planner_context import bind_concentration
    from kaa.idol_road.battle_observation import status_inspection_target
    assert bind_concentration([],bound)[0].value==4
    assert status_inspection_target(bound)==(28,247,31,28)
    assert bind_battle_frame(session(),replace(values,concentration=replace(result,value=99))).observation.concentration is None
    assert bind_battle_frame(session(),replace(values,concentration=replace(result,neutral_reads=()))).observation.concentration is None


@pytest.mark.parametrize('change',['weak_original','weak_other','missing_original','different_proposals'])
def test_neutral_masks_cannot_invent_missing_or_weak_original_proposal(change):
    original,color,gray=interrupted_masks()
    if change=='weak_original':
        original=replace(original,confidence=.97)
    elif change=='weak_other':
        gray=replace(gray,confidence=.97)
    elif change=='missing_original':
        original=replace(original,text=None,confidence=None,row_count=0)
    else:
        gray=replace(gray,text='5',confidence=.97)
    result=confirm_concentration('frame',(28,247,31,28),.99,(original,color,gray),neutral_reads=neutral_reads())
    assert result.value is None


@pytest.mark.parametrize('change',['raw_conflict','raw_invalid','scaled_conflict','neutral_conflict','neutral_invalid'])
def test_all_high_confidence_conflicts_veto_neutral_recheck(change):
    body=list(interrupted_masks())
    neutral=list(neutral_reads())
    scaled=()
    if change=='raw_conflict':
        body[1]=replace(body[1],text='9',confidence=.99)
    elif change=='raw_invalid':
        body[1]=replace(body[1],text='4 1',confidence=.99)
    elif change=='scaled_conflict':
        scaled=tuple(replace(row,text='9',scale=2) for row in reads('9'))
    elif change=='neutral_conflict':
        neutral[0]=replace(neutral[0],text='9')
    else:
        neutral[0]=replace(neutral[0],text='4 1')
    result=confirm_concentration('frame',(28,247,31,28),.99,tuple(body),scaled,tuple(neutral))
    assert result.status=='conflict' and result.value is None


@pytest.mark.parametrize('change',['old_frame','wrong_crop','wrong_scale','duplicate','missing','weak','edge'])
def test_neutral_recheck_keeps_identity_completeness_threshold_and_clipping_guards(change):
    a,b=neutral_reads()
    if change=='old_frame':
        a=replace(a,frame_id='old')
    elif change=='wrong_crop':
        a=replace(a,rect=(59,266,56,24))
    elif change=='wrong_scale':
        a=replace(a,scale=1)
    elif change=='duplicate':
        a=b
    elif change=='weak':
        a=replace(a,confidence=.979)
    elif change=='edge':
        a=replace(a,right_edge_ink=True)
    neutral=(a,) if change=='missing' else (a,b)
    result=confirm_concentration('frame',(28,247,31,28),.99,interrupted_masks(),neutral_reads=neutral)
    assert result.value is None


def enlarged_neutral_seed():
    original,color,gray=reads('4')
    raw=(original,replace(color,confidence=.86),replace(gray,confidence=.91))
    scaled=tuple(replace(row,scale=2) for row in (original,replace(color,text='4mj',confidence=.68),gray))
    return raw,scaled


def test_strong_original_and_enlarged_gray_only_authorize_neutral_recheck():
    raw,scaled=enlarged_neutral_seed()
    assert confirm_concentration('frame',(28,247,31,28),.99,raw,scaled).value is None
    result=confirm_concentration('frame',(28,247,31,28),.99,raw,scaled,neutral_reads())
    assert result.value==4 and result.value_source=='same_crop_neutral_white'


@pytest.mark.parametrize('change',['weak_raw_original','weak_scaled_original','weak_scaled_gray','different_raw','scaled_conflict'])
def test_enlarged_seed_needs_full_raw_proposal_and_independent_strong_agreement(change):
    raw,scaled=enlarged_neutral_seed()
    raw,scaled=list(raw),list(scaled)
    if change=='weak_raw_original':raw[0]=replace(raw[0],confidence=.97)
    elif change=='weak_scaled_original':scaled[0]=replace(scaled[0],confidence=.97)
    elif change=='weak_scaled_gray':scaled[2]=replace(scaled[2],confidence=.97)
    elif change=='different_raw':raw[2]=replace(raw[2],text='5')
    else:scaled[2]=replace(scaled[2],text='5')
    assert confirm_concentration('frame',(28,247,31,28),.99,tuple(raw),tuple(scaled),neutral_reads()).value is None


def expanded_evidence():
    from kaa.idol_road.concentration import ConcentrationExpandedDigits
    from kaa.idol_road.rules import RuleTextRead,RuleTextLine
    body=list(reads('4'));body[1]=replace(body[1],right_edge_ink=True)
    pair=tuple(RuleTextRead('frame',kind,(RuleTextLine('4',.995),)) for kind in ('original','grayscale'))
    return tuple(body),ConcentrationExpandedDigits('frame',body[0].rect,(58,266,130,24),pair,False)


def test_wider_complete_crop_can_independently_confirm_while_preserving_old_clipping_flag():
    raw,proof=expanded_evidence()
    result=confirm_concentration('frame',(28,247,31,28),.99,raw,expanded=proof)
    assert result.value==4 and result.value_source=='expanded_whole_crop'
    assert result.reads[1].right_edge_ink is True
    bound=bind_battle_frame(session(),BattleFrameValues('frame',True,8,40,concentration=result))
    assert bound.observation.concentration==4
    from kaa.idol_road.planner_context import bind_concentration
    assert bind_concentration([],bound)[0].value==4
    for altered in (replace(result,expanded=None),replace(result,value=99)):
        assert bind_battle_frame(session(),BattleFrameValues('frame',True,8,40,concentration=altered)).observation.concentration is None


@pytest.mark.parametrize('change',[
    {'frame_id':'old'}, {'parent_rect':(59,266,56,24)}, {'rect':(59,266,130,24)},
    {'rect':(58,267,130,24)}, {'rect':(58,266,130,23)}, {'rect':(58,266,56,24)},
    {'rect':(58,266,80,24)}, {'rect':(58,266,300,24)}, {'mode':'digit_subset'},
    {'right_edge_ink':True}, {'right_edge_ink':None},
])
def test_expanded_crop_needs_current_source_full_height_real_extension_and_clear_new_boundary(change):
    raw,proof=expanded_evidence()
    assert confirm_concentration('frame',(28,247,31,28),.99,raw,expanded=replace(proof,**change)).value is None


@pytest.mark.parametrize('kind',['weak','missing','stale','conflicting_number','invalid_text','no_raw_seed'])
def test_expanded_readings_cannot_override_strong_conflict_or_invent_missing_raw_seed(kind):
    raw,proof=expanded_evidence()
    a,b=proof.reads
    if kind=='weak':a=replace(a,lines=(replace(a.lines[0],confidence=.979),))
    elif kind=='missing':proof=replace(proof,reads=(a,))
    elif kind=='stale':a=replace(a,frame_id='old')
    elif kind in ('conflicting_number','invalid_text'):
        a=replace(a,lines=(replace(a.lines[0],text='41' if kind=='conflicting_number' else '4%'),))
    else:raw=(replace(raw[0],confidence=.97),*raw[1:])
    if kind!='missing':proof=replace(proof,reads=(a,b))
    assert confirm_concentration('frame',(28,247,31,28),.99,raw,expanded=proof).value is None


def test_complete_wider_pair_can_confirm_when_only_raw_original_was_strong():
    raw,proof=expanded_evidence()
    raw=(raw[0],replace(raw[1],confidence=.96,right_edge_ink=False),replace(raw[2],confidence=.97))
    assert confirm_concentration('frame',(28,247,31,28),.99,raw).value is None
    result=confirm_concentration('frame',(28,247,31,28),.99,raw,expanded=proof)
    assert result.value==4 and result.value_source=='expanded_whole_crop'
    assert result.neutral_reads==()
