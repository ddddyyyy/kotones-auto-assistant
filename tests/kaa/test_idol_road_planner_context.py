from dataclasses import replace
import pytest
from kaa.idol_road.battle_observation import BattleObservationSession, BattleFrameValues, bind_battle_frame
from kaa.idol_road.concentration import ConcentrationNumberRead, confirm_concentration
from kaa.idol_road.planner_context import bind_concentration
from kaa.idol_road.planner_context import IdolRoadBattleContext
from kaa.idol_road.hand_context import CapturedHandContext
from kaa.tasks.produce.new.play_cards.page import LessonBattleContext
from kaa.tasks.produce.new.play_cards.planner.hud_effects import EffectIconObservation


def frame(value=11):
    session=BattleObservationSession('battle','ttmr',18,18,'confirmed_formation_start_request')
    reads=tuple(ConcentrationNumberRead('frame',kind,str(value),.999,(58,266,56,24))
        for kind in ('original','color_white_mask','gray_white_mask'))
    evidence=confirm_concentration('frame',(28,247,31,28),.999,reads)
    return bind_battle_frame(session,BattleFrameValues('frame',True,10,39,concentration=evidence))


def test_skip_recovery_is_lazy_and_unknown_is_cached_within_only_this_frame(monkeypatch):
    from kaa.idol_road.skip_recovery import SkipRecoveryEvidence,confirm_skip_recovery
    calls=[]
    def reader(screen,name):
        calls.append((screen,name))
        return confirm_skip_recovery(SkipRecoveryEvidence(name,False,(644,750,44,38),
            (652,754,26,30),(654,756,23,26),None,None,()))
    monkeypatch.setattr('kaa.idol_road.skip_recovery_page.read_skip_recovery',reader)
    context=IdolRoadBattleContext(None,frame())
    monkeypatch.setattr(context,'fetch_screen',lambda:None)
    assert calls==[]
    first=context.fetch_skip_recovery_observation()
    assert first.status=='unknown' and first.value is None
    assert context.fetch_skip_recovery_observation() is first and len(calls)==1
    context._road_skip_recovery=replace(first,evidence=replace(first.evidence,frame_id='other'))
    assert context.fetch_skip_recovery_observation() is None


def generic(value=211):
    return [EffectIconObservation('lesson_buff',.9,27,244,30,30,value),
            EffectIconObservation('parameter_buff',.9,27,300,30,30,12)]


@pytest.mark.parametrize('hp',[0,39])
def test_current_hp_reuses_this_frame_without_repeating_generic_ocr(monkeypatch,hp):
    current=frame()
    current=replace(current,values=replace(current.values,hp=hp),observation=replace(current.observation,hp=hp))
    def forbidden(*args,**kwargs):
        raise AssertionError('bound HP must not repeat generic OCR')
    monkeypatch.setattr(LessonBattleContext,'fetch_exam_hp',forbidden)
    assert IdolRoadBattleContext(None,current).fetch_exam_hp()==hp


@pytest.mark.parametrize('kind',['unknown','conflict','mismatch','boolean'])
def test_current_hp_invalid_values_stay_unknown_without_ocr_fallback(monkeypatch,kind):
    current=frame()
    if kind=='unknown':
        current=replace(current,values=replace(current.values,hp=None),observation=replace(current.observation,hp=None))
    elif kind=='conflict':
        current=replace(current,observation=replace(current.observation,conflicted_fields=('hp',)))
    elif kind=='mismatch':
        current=replace(current,observation=replace(current.observation,hp=40))
    else:
        current=replace(current,values=replace(current.values,hp=True),observation=replace(current.observation,hp=True))
    monkeypatch.setattr(LessonBattleContext,'fetch_exam_hp',lambda *_:pytest.fail('unknown HP must not be reread'))
    assert IdolRoadBattleContext(None,current).fetch_exam_hp() is None


@pytest.mark.parametrize('kind',['partial','variable_without_round'])
def test_bound_hp_does_not_authorize_incomplete_or_unanchored_frames(kind):
    current=frame()
    if kind=='partial':
        current=replace(current,values=replace(current.values,detail_mode='core_only'))
    else:
        current=replace(current,session=replace(current.session,turn_budget_mode='variable_budget'))
    with pytest.raises(ValueError):
        IdolRoadBattleContext(None,current).fetch_exam_hp()


def test_bound_hp_is_scoped_to_each_frame_instead_of_cached_across_contexts():
    first=frame()
    second=replace(first,values=replace(first.values,frame_id='next',hp=34),
                   observation=replace(first.observation,frame_id='next',hp=34))
    assert IdolRoadBattleContext(None,first).fetch_exam_hp()==39
    assert IdolRoadBattleContext(None,second).fetch_exam_hp()==34


@pytest.mark.parametrize('kind',['valid','stale','conflict'])
def test_bound_hp_revalidates_current_ratio_when_direct_hp_is_unknown(kind):
    from kaa.idol_road.hp_ratio import HpRatioRead,confirm_hp_ratio
    current=frame()
    ratio_frame='old' if kind=='stale' else 'frame'
    ratio=confirm_hp_ratio(ratio_frame,(329,193,47,25),.99,(
        HpRatioRead(ratio_frame,'original','39/53',.99),
        HpRatioRead(ratio_frame,'grayscale','40/53' if kind=='conflict' else '39/53',.99)))
    # Keep the claimed resolved HP deliberately intact to check stale/conflicting
    # raw ratio evidence cannot authorize that claim.
    current=replace(current,values=replace(current.values,hp=None,hp_ratio=ratio))
    assert IdolRoadBattleContext(None,current).fetch_exam_hp()==(39 if kind=='valid' else None)


def test_current_verified_crop_replaces_generic_vote_and_preserves_other_effects():
    observed=generic()
    bound=bind_concentration(observed,frame())
    assert next(item.value for item in bound if item.effect=='lesson_buff')==11
    assert next(item for item in bound if item.effect=='parameter_buff') is observed[1]
    assert observed[0].value==211


def test_verified_zero_and_real_large_values_are_not_clamped():
    for value in (0,211):
        bound=bind_concentration(generic(),frame(value))
        assert next(item.value for item in bound if item.effect=='lesson_buff')==value


@pytest.mark.parametrize('kind',['missing','stale','conflict','forged'])
def test_unknown_or_invalid_evidence_cannot_fall_back_to_generic_211(kind):
    current=frame()
    evidence=current.values.concentration
    if kind=='missing':
        evidence=None
    elif kind=='stale':
        evidence=replace(evidence,frame_id='old')
    elif kind=='forged':
        evidence=replace(evidence,value=211)
    else:
        reads=list(evidence.reads)
        reads[0]=replace(reads[0],text='211')
        evidence=confirm_concentration('frame',evidence.icon_rect,evidence.icon_score,tuple(reads))
    current=replace(current,values=replace(current.values,concentration=evidence))
    assert not any(item.value is not None for item in bind_concentration(generic(),current) if item.effect=='lesson_buff')


@pytest.mark.parametrize('resource',['selection_preview','unconfirmed_overlay'])
def test_predicted_or_occluded_frame_does_not_authorize_planner_observation(resource):
    current=frame()
    with pytest.raises(ValueError):
        bind_concentration(generic(),replace(current,values=replace(current.values,resource_state=resource)))


def test_historical_battle_does_not_authorize_current_planner_values():
    current=frame()
    with pytest.raises(ValueError):
        bind_concentration(generic(),replace(current,session=replace(current.session,identity_source='history')))


def test_unverified_generic_icon_is_not_preserved_as_concentration_presence():
    current=frame()
    missing=replace(current,values=replace(current.values,concentration=None))
    assert not any(item.effect=='lesson_buff' for item in bind_concentration(generic(),missing))


def test_context_effect_cache_is_owned_by_each_frame(monkeypatch):
    import numpy as np
    calls=[]
    def observed(context):
        calls.append(context)
        return generic()
    monkeypatch.setattr(LessonBattleContext,'fetch_effect_observations',observed)
    first=IdolRoadBattleContext(np.zeros((1280,720,3),dtype='uint8'),frame(11))
    second=IdolRoadBattleContext(np.zeros((1280,720,3),dtype='uint8'),frame(19))
    for context,value in ((first,11),(second,19)):
        a=context.fetch_effect_observations()
        b=context.fetch_effect_observations()
        assert a is not b and a==b
        assert next(item.value for item in a if item.effect=='lesson_buff')==value
    assert calls==[first,second]


def test_hand_context_shares_only_same_image_cache_and_preserves_unknown(monkeypatch):
    import numpy as np
    from importlib import import_module
    lesson=import_module('kaa.tasks.produce.new.play_cards.page')
    calls=[]
    def recognized(screen):
        calls.append(screen)
        return [None]
    monkeypatch.setattr(lesson,'locate_cards',recognized)
    screen=np.zeros((1280,720,3),dtype='uint8')
    source=LessonBattleContext(screen)
    assert source.fetch_hands()==[None]
    target=IdolRoadBattleContext(screen,frame(),hand_context=CapturedHandContext('battle','frame',source))
    first=target.fetch_hands()
    second=target.fetch_hands()
    assert first==second==[None] and first is not second
    first.clear()
    assert target.fetch_hands()==[None]
    assert len(calls)==1 and calls[0] is screen


def test_identical_pixels_in_another_image_cannot_reuse_hand_context():
    import numpy as np
    screen=np.zeros((1280,720,3),dtype='uint8')
    with pytest.raises(ValueError,match='不属于当前截图'):
        IdolRoadBattleContext(screen.copy(),frame(),hand_context=CapturedHandContext('battle','frame',LessonBattleContext(screen)))


def test_source_context_cannot_switch_images_after_handoff():
    import numpy as np
    screen=np.zeros((1280,720,3),dtype='uint8')
    source=LessonBattleContext(screen)
    target=IdolRoadBattleContext(screen,frame(),hand_context=CapturedHandContext('battle','frame',source))
    source._screen=screen.copy()
    source._observations.clear()
    with pytest.raises(ValueError,match='不属于当前截图'):
        target.fetch_hands()


def test_foreign_card_object_is_rejected_even_when_context_image_matches(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    screen=np.zeros((1280,720,3),dtype='uint8')
    source=LessonBattleContext(screen)
    monkeypatch.setattr(source,'fetch_hands',lambda:[SimpleNamespace(_screenshot=screen.copy())])
    target=IdolRoadBattleContext(screen,frame(),hand_context=CapturedHandContext('battle','frame',source))
    with pytest.raises(ValueError,match='手牌不属于当前截图'):
        target.fetch_hands()


def test_missing_handoff_preserves_normal_per_context_recognition(monkeypatch):
    import numpy as np
    from importlib import import_module
    lesson=import_module('kaa.tasks.produce.new.play_cards.page')
    calls=[]
    def recognized(screen,*,single_line_titles=False):
        assert single_line_titles
        calls.append(screen)
        return []
    monkeypatch.setattr(lesson,'locate_cards',recognized)
    screen=np.zeros((1280,720,3),dtype='uint8')
    target=IdolRoadBattleContext(screen,frame())
    assert target.fetch_hands()==target.fetch_hands()==[]
    assert len(calls)==1
    other=IdolRoadBattleContext(screen.copy(),frame())
    assert other.fetch_hands()==[] and len(calls)==2


def test_log_hand_cannot_parse_using_another_image_context():
    import numpy as np
    from kaa.idol_road.battle_log_page import read_log_hand
    screen=np.zeros((1280,720,3),dtype='uint8')
    with pytest.raises(ValueError,match='不属于当前截图'):
        read_log_hand(screen,frame(),context=LessonBattleContext(screen.copy()))


@pytest.mark.parametrize('owner,frame_id',[('other','frame'),('battle','other')])
def test_same_image_cannot_reuse_hand_context_for_another_battle_or_frame(owner,frame_id):
    import numpy as np
    screen=np.zeros((1280,720,3),dtype='uint8')
    hand=CapturedHandContext(owner,frame_id,LessonBattleContext(screen))
    with pytest.raises(ValueError,match='不属于当前帧'):
        IdolRoadBattleContext(screen,frame(),hand_context=hand)
