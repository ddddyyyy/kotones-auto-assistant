"""导航请求去重；只比较领域页面身份，不操作设备或推断操作成功。"""

from typing import Literal

from .models import Action, Decision, Observation, Scene


_PAGE_CHANGING_ACTIONS = frozenset((
    Action.OPEN_CONTEST, Action.OPEN_ROAD, Action.SELECT_CHARACTER,
    Action.SELECT_STAGE, Action.SCAN_STAGES, Action.OPEN_FORMATION,
    Action.CONFIRM_FORMATION, Action.BACK_TO_STAGES, Action.BACK_TO_CHARACTERS,
    Action.DISMISS_RESULT, Action.ADVANCE_RESULT, Action.DISMISS_REWARDS,
    Action.CLOSE_INFO,
    Action.CLOSE_BATTLE_STATUS,
    Action.INSPECT_BATTLE_STATUS,
    Action.INSPECT_BATTLE_LOG, Action.CLOSE_BATTLE_LOG,
    Action.OPEN_CONDITIONS, Action.CLOSE_CONDITIONS,
    Action.OPEN_ITEM_DETAILS, Action.CLOSE_ITEM_DETAILS,
    Action.OPEN_SUPPORT, Action.CLOSE_SUPPORT,
    Action.OPEN_AUTO_FORMATION,Action.APPLY_AUTO_FORMATION,Action.CANCEL_AUTO_FORMATION,
    Action.CLOSE_AUTO_FORMATION_RESULT,
))


def _page_identity(observation: Observation) -> tuple[Scene, str | None, int | None]:
    stage = observation.stage
    if observation.scene == Scene.STAGES:
        stage = observation.focused_stage.number if observation.focused_stage is not None else None
    return observation.scene, observation.character_id, stage


class NavigationGuard:
    """发出换页请求后，旧页面只等待；未知画面不抹掉请求记录。"""

    def __init__(self, *, max_waits: int = 3) -> None:
        if max_waits < 1:
            raise ValueError('max_waits must be positive')
        self._max_waits = max_waits
        self._source: tuple[Scene, str | None, int | None] | None = None
        self._action: Action | None = None
        self._waits = 0

    def poll(self, observation: Observation) -> Literal['waiting', 'stalled'] | None:
        if self._source is None or observation.scene == Scene.UNKNOWN:
            return None
        same_map_wait = (
            self._action in (Action.BACK_TO_CHARACTERS, Action.SELECT_STAGE)
            and self._source[0] in (Scene.STAGES, Scene.MAP_OVERVIEW)
            and observation.scene in (Scene.STAGES, Scene.MAP_OVERVIEW)
        )
        # 挑战控件暂失／恢复或焦点变化不证明已离开地图，尤其不能连点返回。
        if not same_map_wait and _page_identity(observation) != self._source:
            self._source = None
            self._action = None
            self._waits = 0
            return None
        self._waits += 1
        return 'stalled' if self._waits >= self._max_waits else 'waiting'

    def record(self, observation: Observation, decision: Decision) -> None:
        memory_transition=(decision.action==Action.REPLACE_SUB_MEMORY and decision.memory_command is not None
                           and decision.memory_command.phase in ('open','apply'))
        if decision.action not in _PAGE_CHANGING_ACTIONS and not memory_transition:
            return
        if decision.action == Action.SELECT_CHARACTER and not observation.character_selectable:
            # 没有候选时执行的是有限列表滚动，不期待场景切换。
            return
        self._source = _page_identity(observation)
        self._action = decision.action
        self._waits = 0
