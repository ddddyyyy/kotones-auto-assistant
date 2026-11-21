"""Current extra-play control evidence; never a simulated remaining action."""
from dataclasses import dataclass
from typing import Literal
from dataclasses import replace
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .battle_observation import BattleFrameObservation
    from .battle_profile import SenseBattleProfile
    from kaa.tasks.produce.new.play_cards.planner.state import BattleState


@dataclass(frozen=True)
class ExtraPlayObservation:
    frame_id: str
    rect: tuple[int,int,int,int]
    one_score: float
    stroke_coverage: float
    unique_one: bool
    value: int | None
    scope: Literal['absent_or_literal_one','absent_or_literal_one_or_two'] = 'absent_or_literal_one'
    method: Literal['complete_control_masked_correlation'] = 'complete_control_masked_correlation'
    blue_coverage: float | None=None
    one_blue_coverage: float | None=None
    two_score: float | None=None
    two_stroke_coverage: float | None=None
    unique_two: bool | None=None
    two_blue_coverage: float | None=None


def confirm_extra_play(frame_id: str,rect: tuple[int,int,int,int],one_score: float,
                        stroke_coverage: float,unique_one: bool,blue_coverage: float | None=None,
                        one_blue_coverage: float | None=None,*,two_score: float | None=None,
                        two_stroke_coverage: float | None=None,unique_two: bool | None=None,
                        two_blue_coverage: float | None=None) -> ExtraPlayObservation:
    value=None
    x,y,w,h=rect
    two_present=any(v is not None for v in (two_score,two_stroke_coverage,unique_two,two_blue_coverage))
    two_valid=(type(unique_two) is bool and all(isinstance(v,(int,float)) and not isinstance(v,bool) and 0<=v<=1
        for v in (two_score,two_stroke_coverage,two_blue_coverage)))
    if (frame_id and min(rect)>=0 and min(w,h)>0 and x+w<=720 and y+h<=1280
        and 0<=one_score<=1 and 0<=stroke_coverage<=1 and type(unique_one) is bool):
        blue_known=(isinstance(blue_coverage,(int,float)) and not isinstance(blue_coverage,bool)
            and isinstance(one_blue_coverage,(int,float)) and not isinstance(one_blue_coverage,bool)
            and 0<=blue_coverage<=1 and 0<=one_blue_coverage<=1)
        legacy=blue_coverage is None and one_blue_coverage is None
        if (.97<=one_score<=1 and unique_one and stroke_coverage>=.95
            and (legacy or (blue_known and isinstance(one_blue_coverage,(int,float)) and one_blue_coverage>=.6))):
            value=1
        elif ((legacy or blue_known) and stroke_coverage<=.05 and one_score<.97) or (
            blue_known and isinstance(blue_coverage,(int,float)) and blue_coverage<=.05):
            value=0
        if two_present:
            if not two_valid or not blue_known:value=None
            elif (two_score is not None and two_stroke_coverage is not None and two_blue_coverage is not None
                and two_score>=.97 and two_stroke_coverage>=.95 and unique_two and two_blue_coverage>=.6):
                value=None if value==1 else 2
            elif value==0 and not (blue_coverage is not None and blue_coverage<=.05):
                if two_stroke_coverage is None or two_stroke_coverage>.05:value=None
    return ExtraPlayObservation(frame_id,rect,one_score,stroke_coverage,unique_one,value,
        scope='absent_or_literal_one_or_two' if two_present else 'absent_or_literal_one',
        blue_coverage=blue_coverage,one_blue_coverage=one_blue_coverage,two_score=two_score,
        two_stroke_coverage=two_stroke_coverage,unique_two=unique_two,two_blue_coverage=two_blue_coverage)


def extra_play_valid(evidence: ExtraPlayObservation,frame_id: str) -> bool:
    return (evidence.frame_id==frame_id and confirm_extra_play(evidence.frame_id,evidence.rect,
        evidence.one_score,evidence.stroke_coverage,evidence.unique_one,
        evidence.blue_coverage,evidence.one_blue_coverage,two_score=evidence.two_score,
        two_stroke_coverage=evidence.two_stroke_coverage,unique_two=evidence.unique_two,
        two_blue_coverage=evidence.two_blue_coverage)==evidence)


def bind_sense_play_count(state: 'BattleState',frame: 'BattleFrameObservation',
                          evidence: ExtraPlayObservation | None,profile: 'SenseBattleProfile') -> 'BattleState | None':
    """Use this Sense battle's base action plus its current visible extras."""
    from .battle_observation import current_absolute_turn
    from kaa.db.constants import ProducePlanType
    if (evidence is None or not extra_play_valid(evidence,frame.values.frame_id)
        or type(evidence.value) is not int or evidence.value not in (0,1,2)
        or profile.session!=frame.session or profile.plan_type!=ProducePlanType.Plan1
        or profile.source!='confirmed_formation_start_recommendation'
        or not profile.formation_frame_id or state.is_exam is not True
        or current_absolute_turn(frame) is None or state.absolute_turn_observation!=current_absolute_turn(frame)
        or state.remaining_turns!=frame.values.remaining_turns or not state.hp_known or state.hp!=frame.observation.hp):
        return None
    return replace(state,playable_count=1+evidence.value,plan_type=profile.plan_type)
