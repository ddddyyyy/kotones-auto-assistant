"""单帧页面观察与流程命令；未知状态不作为通关证据。"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .log_hand import LogHandObservation
    from .skip_probe import SkipLogResolution
    from .card_probe import CardLogResolution
    from .round_anchor import RoundAnchorState
    from .battle_inputs import BattleInputState
    from .battle_log import LogInspection, LogReturn
    from .rule_conditions import RuleConditionObservation
    from .status_details import BattleStatusDetails
    from .formation import FormationSlots
    from .memory_picker import SubMemoryViewport
    from .memory_replacement import MemoryCommand

from .objectives import StarRequirement
from .items import ItemViewport, VisibleItem
from .rules import RuleViewport


class Scene(str, Enum):
    UNKNOWN = 'unknown'
    HOME = 'home'
    RESUME_PROMPT = 'resume_prompt'
    """游戏保留未完成挑战；不据此恢复程序会话或推断关卡身份。"""
    CONTEST_HUB = 'contest_hub'
    CHARACTERS = 'characters'
    STAGES = 'stages'
    MAP_OVERVIEW = 'map_overview'
    """地图身份已确认，但当前没有可用挑战／复位控件；不代表锁定或全通关。"""
    DETAIL = 'detail'
    FORMATION = 'formation'
    CONFIRM = 'confirm'
    AUTO_FORMATION = 'auto_formation'
    AUTO_FORMATION_RESULT = 'auto_formation_result'
    SUB_MEMORY_SELECTOR = 'sub_memory_selector'
    BATTLE = 'battle'
    BATTLE_LOG = 'battle_log'
    BATTLE_STATUS_DETAILS = 'battle_status_details'
    JUDGING = 'judging'
    RESULT = 'result'
    REWARDS = 'rewards'
    REWARD_INFO = 'reward_info'
    CONDITIONS = 'conditions'
    ITEM_DETAILS = 'item_details'
    SUPPORT_DETAILS = 'support_details'


class Outcome(str, Enum):
    SUCCESS = 'success'
    FAILURE = 'failure'
    UNKNOWN = 'unknown'


class Action(str, Enum):
    WAIT = 'wait'
    OPEN_CONTEST = 'open_contest'
    OPEN_ROAD = 'open_road'
    SELECT_CHARACTER = 'select_character'
    SELECT_STAGE = 'select_stage'
    NEXT_STAGE = 'next_stage'
    BACK_TO_STAGES = 'back_to_stages'
    SCAN_STAGES = 'scan_stages'
    OPEN_FORMATION = 'open_formation'
    CONFIRM_FORMATION = 'confirm_formation'
    OPEN_AUTO_FORMATION = 'open_auto_formation'
    APPLY_AUTO_FORMATION = 'apply_auto_formation'
    CANCEL_AUTO_FORMATION = 'cancel_auto_formation'
    CLOSE_AUTO_FORMATION_RESULT = 'close_auto_formation_result'
    CANCEL_MEMORY_SELECTOR = 'cancel_memory_selector'
    REPLACE_SUB_MEMORY = 'replace_sub_memory'
    START_BATTLE = 'start_battle'
    INSPECT_BATTLE_HP = 'inspect_battle_hp'
    CLOSE_BATTLE_HP = 'close_battle_hp'
    INSPECT_BATTLE_LOG = 'inspect_battle_log'
    CLOSE_BATTLE_LOG = 'close_battle_log'
    SCROLL_BATTLE_LOG = 'scroll_battle_log'
    SCROLL_BATTLE_LOG_HEADER = 'scroll_battle_log_header'
    INSPECT_BATTLE_STATUS = 'inspect_battle_status'
    CLOSE_BATTLE_STATUS = 'close_battle_status'
    SCROLL_BATTLE_STATUS = 'scroll_battle_status'
    DISMISS_RESULT = 'dismiss_result'
    ADVANCE_RESULT = 'advance_result'
    DISMISS_REWARDS = 'dismiss_rewards'
    CLOSE_INFO = 'close_info'
    OPEN_CONDITIONS = 'open_conditions'
    CLOSE_CONDITIONS = 'close_conditions'
    OPEN_ITEM_DETAILS = 'open_item_details'
    CLOSE_ITEM_DETAILS = 'close_item_details'
    SCROLL_ITEMS = 'scroll_items'
    OPEN_SUPPORT = 'open_support'
    CLOSE_SUPPORT = 'close_support'
    SCROLL_SUPPORT = 'scroll_support'
    BACK_TO_CHARACTERS = 'back_to_characters'
    READY = 'ready'
    STOP = 'stop'


@dataclass(frozen=True)
class Stage:
    number: int
    unlocked: bool
    stars: int | None

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError('stage number must be positive')
        validate_stars(self.stars)

    @property
    def cleared(self) -> bool | None:
        return None if self.stars is None else self.stars >= 1


def validate_stars(stars: int | None) -> None:
    if stars is not None and (type(stars) is not int or not 0 <= stars <= 3):
        raise ValueError('stars must be an integer from 0 to 3, or None')


@dataclass(frozen=True)
class Observation:
    scene: Scene
    character_id: str | None = None
    stage: int | None = None
    stages: tuple[Stage, ...] = ()
    stages_complete: bool = False
    """True 仅在完整列表扫描已完成时成立，空白帧不代表全通关。"""
    outcome: Outcome = Outcome.UNKNOWN
    formation_ready: bool = False
    character_selectable: bool = False
    """角色列表中已确认配置角色的可点击姓名；不证明该角色已被选中。"""
    focused_stage: Stage | None = None
    """地图当前聚焦的关卡；不表示已扫描完整列表或证明最早未通关。"""
    stars: int | None = None
    result_next_available: bool = False
    """当前页面明确识别的关卡星数；不是标题、名次或历史星数的推断。"""
    conditions_available: bool = False
    requirements: tuple[StarRequirement, ...] = ()
    """当前条件弹窗的文字／星档；描述目标，不代表获得星数。"""
    items_available: bool = False
    visible_items: tuple[VisibleItem, ...] | None = None
    item_viewport: ItemViewport | None = None
    support_available: bool = False
    rule_viewport: RuleViewport | None = None
    total_turns: int | None = None
    battle_state: 'RuleConditionObservation | None' = None
    battle_hp_info_available: bool = False
    battle_log_available: bool = False
    battle_log_pending: bool = False
    battle_input_state: 'BattleInputState | None' = None
    battle_round_anchor: 'RoundAnchorState | None' = None
    battle_skip_resolution: 'SkipLogResolution | None' = None
    battle_card_resolution: 'CardLogResolution | None' = None
    battle_log_hand: 'LogHandObservation | None' = None
    battle_log_inspection: 'LogInspection | None' = None
    battle_log_return: 'LogReturn | None' = None
    battle_status_details: 'BattleStatusDetails | None' = None
    battle_status_info_available: bool = False
    battle_status_scroll_available: bool = False
    formation_slots: 'FormationSlots | None' = None
    auto_formation_available: bool = False
    auto_formation_confirm_available: bool = False
    auto_formation_result_pending: bool = False
    formation_recommended_effect: str | None = None
    memory_selector: 'SubMemoryViewport | None' = None
    memory_replacement_token: str | None = None
    memory_replacement_receipt: str | None = None

    def __post_init__(self) -> None:
        validate_stars(self.stars)


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    character_id: str | None = None
    stage: int | None = None
    memory_command: 'MemoryCommand | None' = None
