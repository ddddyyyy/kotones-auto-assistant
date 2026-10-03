"""游戏内置自动出牌评价数据访问。

这些表来自运行时下载的 ``game.db``。卡牌专用评价优先于通用评价；
调用方不需要了解具体表结构，也不会在旧版数据库缺表时中断培育。
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from functools import lru_cache

from kaa.db._util import register_cache_clear, row_dict
from kaa.db.sqlite import select

logger = logging.getLogger(__name__)

AUTO_PLAY_TYPE = 'ExamPlayType_AutoPlay'
MIN_REMAINING_TERM = 1
MAX_REMAINING_TERM = 7


@dataclass(frozen=True)
class AutoPlayCardEvaluation:
    """一张卡在指定剩余回合的游戏内置评价。"""

    card_id: str
    remaining_term: int
    evaluation: int
    play_type: str | None

    @property
    def is_override(self) -> bool:
        """是否来自特定玩法的卡牌覆盖表。"""
        return self.play_type is not None


@dataclass(frozen=True)
class AutoEffectEvaluation:
    """Official effect weight for one archetype and remaining-turn bucket."""

    exam_effect_type: str
    remaining_term: int
    evaluation_type: str
    evaluation: int
    status_enchant_coefficient_permil: int


@dataclass(frozen=True)
class AutoTriggerEvaluation:
    trigger_id: str
    coefficient_permil: int
    count: int


def normalize_remaining_term(remaining_term: int | None) -> int:
    """将 UI 回合数映射到评价表支持的 1～7 回合。"""
    if remaining_term is None:
        return MAX_REMAINING_TERM
    return max(MIN_REMAINING_TERM, min(MAX_REMAINING_TERM, remaining_term))


@lru_cache(maxsize=4096)
def get_auto_play_card_evaluation(
    card_id: str,
    remaining_term: int | None,
    play_type: str = AUTO_PLAY_TYPE,
) -> AutoPlayCardEvaluation | None:
    """查询卡牌评价，优先使用玩法专用覆盖值。

    旧版 ``game.db`` 可能没有自动评价表。此时返回 ``None``，让上层策略
    显式降级到游戏推荐，而不是让整场培育失败。
    """
    term = normalize_remaining_term(remaining_term)
    try:
        row = select(
            """
            SELECT type, produceCardId, remainingTerm, evaluation
            FROM ProduceExamAutoPlayProduceCardEvaluation
            WHERE type = ? AND produceCardId = ? AND remainingTerm = ?
            LIMIT 1;
            """,
            play_type,
            card_id,
            term,
        )
        if row is not None:
            data = row_dict(row)
            return AutoPlayCardEvaluation(
                card_id=data['produceCardId'],
                remaining_term=data['remainingTerm'],
                evaluation=data['evaluation'],
                play_type=data['type'],
            )

        row = select(
            """
            SELECT produceCardId, remainingTerm, evaluation
            FROM ProduceExamAutoPlayCardEvaluation
            WHERE produceCardId = ? AND remainingTerm = ?
            LIMIT 1;
            """,
            card_id,
            term,
        )
    except sqlite3.OperationalError:
        logger.warning('Auto-play evaluation tables are unavailable.', exc_info=True)
        return None

    if row is None:
        return None
    data = row_dict(row)
    return AutoPlayCardEvaluation(
        card_id=data['produceCardId'],
        remaining_term=data['remainingTerm'],
        evaluation=data['evaluation'],
        play_type=None,
    )


@lru_cache(maxsize=4096)
def get_auto_effect_evaluation(
    exam_effect_type: str,
    remaining_term: int | None,
    evaluation_type: str,
    play_type: str = AUTO_PLAY_TYPE,
) -> AutoEffectEvaluation | None:
    """Read an official effect weight without exposing the master-data schema."""
    term = normalize_remaining_term(remaining_term)
    try:
        row = select(
            """
            SELECT examEffectType, remainingTerm, evaluationType, evaluation,
                   examStatusEnchantCoefficientPermil
            FROM ProduceExamAutoEvaluation
            WHERE type = ? AND examEffectType = ? AND remainingTerm = ?
              AND evaluationType = ?
            LIMIT 1;
            """,
            play_type,
            exam_effect_type,
            term,
            evaluation_type,
        )
    except sqlite3.OperationalError:
        logger.warning('Auto effect evaluation table is unavailable.', exc_info=True)
        return None
    if row is None:
        return None
    data = row_dict(row)
    return AutoEffectEvaluation(
        exam_effect_type=data['examEffectType'],
        remaining_term=data['remainingTerm'],
        evaluation_type=data['evaluationType'],
        evaluation=data['evaluation'],
        status_enchant_coefficient_permil=(
            data['examStatusEnchantCoefficientPermil']
        ),
    )


@lru_cache(maxsize=2048)
def get_auto_trigger_evaluation(
    trigger_id: str,
    play_type: str = AUTO_PLAY_TYPE,
) -> AutoTriggerEvaluation | None:
    """Read the official expected-value coefficient for an enchant trigger."""
    if not trigger_id:
        return None
    try:
        row = select(
            """
            SELECT examStatusEnchantProduceExamTriggerId, coefficientPermil, count
            FROM ProduceExamAutoTriggerEvaluation
            WHERE type = ? AND examStatusEnchantProduceExamTriggerId = ?
            LIMIT 1;
            """,
            play_type,
            trigger_id,
        )
    except sqlite3.OperationalError:
        logger.warning('Auto trigger evaluation table is unavailable.', exc_info=True)
        return None
    if row is None:
        return None
    return AutoTriggerEvaluation(
        trigger_id=row['examStatusEnchantProduceExamTriggerId'],
        coefficient_permil=row['coefficientPermil'],
        count=row['count'],
    )


register_cache_clear(get_auto_play_card_evaluation.cache_clear)
register_cache_clear(get_auto_effect_evaluation.cache_clear)
register_cache_clear(get_auto_trigger_evaluation.cache_clear)
