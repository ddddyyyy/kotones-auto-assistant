from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from kaa.db.skill_card import SkillCard
    from .page import LessonBattleContext

class AbstractBattleStrategy(ABC):
    """考试或课程时出牌策略"""
    @abstractmethod
    def on_action(self, ctx: 'LessonBattleContext') -> bool:
        """执行出牌动作。

        :returns: 返回 True 表示已处理，False 表示未处理。如果未处理，将会走兜底推荐卡检测逻辑。
        """
        pass

    def on_battle_end(self) -> None:
        """Release pending diagnostics when the battle loop ends normally."""
        pass

    def rank_move_cards(self, cards: list['SkillCard | None']) -> list[int] | None:
        """Order cards for move-to-hold; None preserves the legacy fallback."""
        return None
