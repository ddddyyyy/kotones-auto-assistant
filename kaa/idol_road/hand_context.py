"""同一战斗帧内的手牌识别交接；不跨帧保存或授予输入权限。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from cv2.typing import MatLike
if TYPE_CHECKING:
    from .battle_observation import BattleFrameObservation
    from kaa.tasks.produce.new.play_cards.page import LessonBattleContext
    from kaa.tasks.produce.new.play_cards.ui import CardGameObject


@dataclass(frozen=True)
class CapturedHandContext:
    battle_id: str
    frame_id: str
    context: LessonBattleContext

    def validate(self,screen: MatLike,frame: BattleFrameObservation) -> None:
        if (not self.battle_id or not self.frame_id or self.battle_id!=frame.session.battle_id
            or self.frame_id!=frame.values.frame_id):
            raise ValueError('手牌Context不属于当前帧')
        if self.context.fetch_screen() is not screen:
            raise ValueError('手牌Context不属于当前截图')

    def read(self,screen: MatLike,frame: BattleFrameObservation) -> list[CardGameObject | None]:
        self.validate(screen,frame)
        hands=self.context.fetch_hands()
        if any(card is not None and card._screenshot is not screen for card in hands):
            raise ValueError('手牌不属于当前截图')
        return list(hands)
