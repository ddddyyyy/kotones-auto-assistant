from dataclasses import replace
import pytest
from kaa.db.constants import ProduceExamEffectType, ProducePlanType
from kaa.idol_road.battle_profile import sense_profile_from_start
from kaa.idol_road.battle_observation import BattleObservationSession
from kaa.idol_road.models import Scene
from kaa.idol_road.page import IdolRoadPage
from tests.kaa.test_idol_road_formation import formation, slots

SESSION = BattleObservationSession('owned', 'ttmr', 19, 20, 'confirmed_formation_start_request', 'variable_budget')


@pytest.mark.parametrize('recommendation,kind', [('concentration', ProduceExamEffectType.ExamLessonBuff),
    ('good_condition', ProduceExamEffectType.ExamParameterBuff)])
def test_profile_is_observed_at_owned_start_and_keeps_formation_source(recommendation, kind):
    obs = formation(slots(4, False), formation_recommended_effect=recommendation)
    profile = sense_profile_from_start(SESSION, obs)
    assert profile.session == SESSION and profile.archetype == kind and profile.plan_type == ProducePlanType.Plan1
    assert profile.formation_frame_id == obs.formation_slots.frame_id


@pytest.mark.parametrize('change', [{'scene': Scene.BATTLE}, {'formation_ready': False},
    {'formation_slots': None}, {'formation_slots': slots()}, {'character_id': 'other'}, {'stage': 20},
    {'formation_recommended_effect': None}, {'formation_recommended_effect': 'anomaly'}])
def test_unknown_or_unrelated_formation_does_not_invent_a_sense_profile(change):
    obs = formation(slots(4, False), formation_recommended_effect='good_condition')
    assert sense_profile_from_start(SESSION, replace(obs, **change)) is None


def test_old_profile_is_inaccessible_after_new_battle_or_owner_loss():
    page = object.__new__(IdolRoadPage)
    page._battle_session = SESSION
    page._battle_sense_profile = sense_profile_from_start(SESSION, formation(slots(4, False), formation_recommended_effect='good_condition'))
    assert page.battle_sense_profile is not None
    page._battle_session = replace(SESSION, battle_id='new')
    assert page.battle_sense_profile is None
    page._battle_session = None
    assert page.battle_sense_profile is None
