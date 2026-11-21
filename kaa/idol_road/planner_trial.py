"""单场实验性Planner输入与运行边界；不将部分准备报告标记成自动就绪。"""
from dataclasses import dataclass
from typing import Literal
from kaa.db.constants import ProduceExamEffectType
from .models import Observation,Scene
from .objectives import StageObjectives
from .rules import StageRules
from .turn_budget import fixed_turn_budget_conflicts


class UnsupportedTurnBudget(ValueError):
    """可见追加回合尚未适配，不能用固定总回合实验策略开战。"""


@dataclass(frozen=True)
class PlannerTrialSpec:
    character_id: str
    stage: int
    rank_at_most: int
    total_turns: int
    archetype: ProduceExamEffectType = ProduceExamEffectType.ExamLessonBuff
    scope: Literal['single_concentration_trial', 'single_good_condition_trial'] = 'single_concentration_trial'


def prepare_planner_trial(observation: Observation, objectives: StageObjectives | None,
                           stage_rules: StageRules | None = None) -> PlannerTrialSpec:
    if (observation.formation_recommended_effect not in (None,'concentration','good_condition')
        or (observation.formation_slots is not None and observation.formation_recommended_effect is None)):
        raise ValueError('当前编组的推荐效果不是已确认的集中或好调，不能选择实验策略')
    if observation.formation_recommended_effect == 'good_condition' and observation.formation_slots is None:
        raise ValueError('好调试跑需要当前层完整槽位报告')
    if observation.formation_slots is not None and not observation.formation_slots.full:
        raise ValueError('本层编组槽位未确认填满，不能进行AI试跑')
    if (observation.scene!=Scene.CONFIRM or observation.character_id!='ttmr'
        or type(observation.stage) is not int or observation.stage<1
        or observation.stars!=0 or not observation.formation_ready):
        raise ValueError('AI单场试运行只支持手毬的已确认零星现有编成')
    if (objectives is None or (objectives.character_id,objectives.stage)!=('ttmr',observation.stage)
        or objectives.identity_source!='opened_from_confirmed_formation'
        or objectives.source!='conditions_dialog'):
        raise ValueError('AI试运行需要当前编成读取的一星目标')
    rank=objectives.rank_limit(1)
    total=observation.total_turns
    if type(rank) is not int or not 1<=rank<=6 or type(total) is not int or not 1<=total<=30:
        raise ValueError('AI试运行的一星名次目标或总回合未知')
    if fixed_turn_budget_conflicts('ttmr',observation.stage,stage_rules):
        raise UnsupportedTurnBudget('当前场地含追加回合，固定总回合试跑尚未适配；未发出挑战')
    if observation.formation_recommended_effect == 'good_condition':
        return PlannerTrialSpec('ttmr',observation.stage,rank,total,ProduceExamEffectType.ExamParameterBuff,
                                 'single_good_condition_trial')
    return PlannerTrialSpec('ttmr',observation.stage,rank,total)
