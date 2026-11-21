"""只读当前完整手牌的禁用状态及启用SKIP文字，不读取元气冒充次数。"""
import cv2
from cv2.typing import MatLike
from kotonebot.backend.ocr import en,jp
from kaa.tasks import R
from .turn_end import HandCardState, hand_is_blocked


def read_turn_end(screen: MatLike) -> tuple[bool, bool, tuple[HandCardState, ...], int]:
    from kaa.tasks.produce.new.play_cards.page import LessonBattleContext
    from kaa.tasks.produce.shared.cards import skill_card_count
    count = skill_card_count(screen)
    context = LessonBattleContext(screen)
    hands = context.fetch_hands()
    same_frame = context.fetch_screen() is screen
    cards = tuple(HandCardState(
        hand.card._id if hand is not None and hand.card is not None else None,
        hand.available if hand is not None and same_frame else None,
        (hand.rect.x1, hand.rect.y1, hand.rect.w, hand.rect.h) if hand is not None else None,
    ) for hand in hands)
    empty = count == 0 and not cards and same_frame and read_empty_hand(screen)
    return hand_is_blocked(count, cards) or empty, read_enabled_skip(screen), cards, count


def read_empty_hand(screen: MatLike) -> bool:
    """完整零手牌提示才是证据；没有识别出卡牌本身不算空手。"""
    box=R.IdolRoad.BoxEmptyHandMessage
    crop=screen[box.y1:box.y2,box.x1:box.x2]
    for variant in (crop,cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)):
        rows=jp().ocr(variant)
        if (len(rows)!=1 or not .98<=rows[0].confidence<=1
            or rows[0].text!='手札のスキルカードが0枚です'):
            return False
    return True


def read_enabled_skip(screen: MatLike) -> bool:
    box = R.IdolRoad.BoxEnabledSkipText
    crop = screen[box.y1:box.y2, box.x1:box.x2]
    # 启用按钮是白底黑字；禁用灰底和动画淡字不当作可点击。
    enabled = bool((crop.min(axis=2) > 230).mean() > .5)
    if enabled:
        for variant in (crop, cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)):
            rows = en().ocr(variant)
            if len(rows) != 1 or rows[0].confidence < .98 or rows[0].text != 'SKIP':
                enabled = False
                break
    return enabled
