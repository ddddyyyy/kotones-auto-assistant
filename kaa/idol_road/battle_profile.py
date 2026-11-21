"""The owned START's observed Sense recommendation; no execution permission."""
from dataclasses import dataclass
from typing import Literal
from kaa.db.constants import ProduceExamEffectType, ProducePlanType
from .battle_observation import BattleObservationSession
from .models import Observation, Scene


@dataclass(frozen=True)
class SenseBattleProfile:
    session: BattleObservationSession
    formation_frame_id: str
    archetype: ProduceExamEffectType
    plan_type: ProducePlanType = ProducePlanType.Plan1
    source: Literal['confirmed_formation_start_recommendation'] = 'confirmed_formation_start_recommendation'


def sense_profile_from_start(session: BattleObservationSession, observation: Observation) -> SenseBattleProfile | None:
    """Called by Page at its actual owned START, not from later battle imagery."""
    slots = observation.formation_slots
    kinds = {'concentration': ProduceExamEffectType.ExamLessonBuff,
             'good_condition': ProduceExamEffectType.ExamParameterBuff}
    kind = kinds.get(observation.formation_recommended_effect or '')
    if (kind is None or observation.scene != Scene.CONFIRM or not observation.formation_ready
        or slots is None or not slots.full or session.identity_source != 'confirmed_formation_start_request'
        or not session.battle_id or (observation.character_id, observation.stage) != (session.character_id, session.stage)):
        return None
    return SenseBattleProfile(session, slots.frame_id, kind)
