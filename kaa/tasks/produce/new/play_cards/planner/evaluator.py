"""第一阶段规划器：使用游戏内置卡牌评价作为可解释先验。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Protocol

from kaa.db.constants import ProduceExamEffectType, ProducePlanType
from kaa.db.exam_status_enchant import get_exam_status_enchant
from kaa.db.produce_exam_auto import (
    get_auto_effect_evaluation,
    get_auto_play_card_evaluation,
    get_auto_trigger_evaluation,
)
from kaa.db.skill_card import ProduceExamEffect, SkillCard

from .state import BattleState
from .profiles import profile_for
from .costs import play_cost
from .simulator import simulate_play
from .trigger import TriggerVerdict, evaluate_trigger_id, load_exam_trigger


class EvaluationConfidence(str, Enum):
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'


@dataclass(frozen=True)
class CardEvaluation:
    card_id: str
    score: float
    confidence: EvaluationConfidence
    reason: str
    projected_output: float = 0.0
    confirmed_output: float = 0.0


def conservative_target_output(confirmed_output: float) -> int:
    """Conservative integer estimate for hard targets, not expected value.

    Small fractional output can disappear under the game's intermediate
    integer rounding. One live play predicted 1.125 but moved no visible
    lesson progress, so sub-two estimates must not assert target completion.
    """
    if confirmed_output < 2.0:
        return 0
    return max(0, math.floor(confirmed_output))


class CardEvaluator(Protocol):
    """规划器可替换的卡牌评价接口。"""

    def evaluate(self, card: SkillCard, state: BattleState) -> CardEvaluation | None:
        ...


class OfficialAutoEvaluator:
    """按官方评价表对候选卡进行一步评价。

    当前阶段只负责可靠地接入数据，不伪造尚未实现的效果模拟。后续模拟器的
    状态增量会叠加到该先验上。
    """

    def evaluate(self, card: SkillCard, state: BattleState) -> CardEvaluation | None:
        row = get_auto_play_card_evaluation(card._id, state.remaining_turns)
        if row is None:
            return None
        confidence = (
            EvaluationConfidence.HIGH
            if row.is_override
            else EvaluationConfidence.MEDIUM
        )
        source = '玩法专用评价' if row.is_override else '通用卡牌评价'
        return CardEvaluation(
            card_id=card._id,
            score=float(row.evaluation),
            confidence=confidence,
            reason=(
                f'{source}，评价表索引 {row.remaining_term}，'
                + (f'剩余 {state.remaining_turns} 回合'
                   if state.remaining_turns_known else '剩余回合未知，使用保守窗口')
            ),
        )


class EffectHeuristicEvaluator:
    """Score immediate output, survival and setup without reusing old rules."""

    _DIRECT = {ProduceExamEffectType.ExamLesson, ProduceExamEffectType.ExamLessonFix}
    _DEFENCE = {
        ProduceExamEffectType.ExamBlock,
        ProduceExamEffectType.ExamBlockFix,
        ProduceExamEffectType.ExamStaminaRecoverFix,
    }
    _SETUP = {
        ProduceExamEffectType.ExamLessonBuff,
        ProduceExamEffectType.ExamParameterBuff,
        ProduceExamEffectType.ExamConcentration,
        ProduceExamEffectType.ExamCardPlayAggressive,
    }
    _OFFICIAL_EVALUATION_TYPES = {
        ProduceExamEffectType.ExamLessonBuff: 'ExamLessonBuff',
        ProduceExamEffectType.ExamLessonBuffAdditive: 'ExamLessonBuffAdditive',
        ProduceExamEffectType.ExamParameterBuff: 'ExamParameterBuff',
        ProduceExamEffectType.ExamParameterBuffAdditive: (
            'ExamParameterBuffAdditive'
        ),
        ProduceExamEffectType.ExamParameterBuffMultiplePerTurn: (
            'ParameterBuffMultiplePerTurn'
        ),
        ProduceExamEffectType.ExamReview: 'ExamReview',
        ProduceExamEffectType.ExamReviewAdditive: 'ExamReviewAdditive',
        ProduceExamEffectType.ExamReviewMultiple: 'ExamReviewMultiple',
        ProduceExamEffectType.ExamCardPlayAggressive: 'ExamCardPlayAggressive',
        ProduceExamEffectType.ExamAggressiveAdditive: 'ExamAggressiveAdditive',
        ProduceExamEffectType.ExamConcentration: 'ExamConcentration',
        ProduceExamEffectType.ExamPreservation: 'ExamPreservation',
        ProduceExamEffectType.ExamFullPowerPoint: 'ExamFullPowerPointAdditive',
        ProduceExamEffectType.ExamFullPowerPointAdditive: (
            'ExamFullPowerPointAdditive'
        ),
        ProduceExamEffectType.ExamFullPower: 'ExamFullPower',
        ProduceExamEffectType.ExamEnthusiasticAdditive: (
            'ExamEnthusiasticAdditive'
        ),
        ProduceExamEffectType.ExamEnthusiasticMultiple: 'ExamEnthusiasticMultiple',
        ProduceExamEffectType.ExamPlayableValueAdd: 'PlayableValueAdd',
        ProduceExamEffectType.ExamExtraTurn: 'ExamExtraTurn',
        ProduceExamEffectType.ExamAntiDebuff: 'ExamAntiDebuff',
        ProduceExamEffectType.ExamBlockRestriction: 'ExamBlockRestriction',
        ProduceExamEffectType.ExamStaminaConsumptionAdd: (
            'ExamStaminaConsumptionAdd'
        ),
        ProduceExamEffectType.ExamStaminaConsumptionAddFix: (
            'ExamStaminaConsumptionAdd'
        ),
        ProduceExamEffectType.ExamStaminaConsumptionDown: (
            'ExamStaminaConsumptionDown'
        ),
        ProduceExamEffectType.ExamStaminaConsumptionDownFix: (
            'ExamStaminaConsumptionDownFix'
        ),
        ProduceExamEffectType.ExamLessonValueMultiple: 'ExamLessonValueMultiple',
        ProduceExamEffectType.ExamLessonValueMultipleDependReviewOrAggressive: (
            'ExamLessonValueMultipleDependReviewOrAggressive'
        ),
        ProduceExamEffectType.StanceLock: 'StanceLock',
    }
    _STATUS_ENCHANT_EVALUATION_TYPES = {
        ProduceExamEffectType.ExamLesson: 'Parameter',
        ProduceExamEffectType.ExamLessonFix: 'Parameter',
        ProduceExamEffectType.ExamBlock: 'Block',
        ProduceExamEffectType.ExamBlockFix: 'Block',
        ProduceExamEffectType.ExamStaminaRecoverFix: 'Stamina',
        ProduceExamEffectType.ExamCardDraw: 'DrawCardCount',
    }

    def evaluate(self, card: SkillCard, state: BattleState) -> CardEvaluation:
        turns = max(1, state.remaining_turns)
        early_weight = min(6, turns)
        late_weight = 1.2 if turns <= 2 else 1.0
        score = 0.0
        projected_output = 0.0
        confirmed_output = 0.0
        reasons: list[str] = []
        profile = profile_for(state.archetype)

        try:
            card_plan = ProducePlanType(card.plan_type) if card.plan_type else None
        except ValueError:
            card_plan = None
        cross_plan = (
            profile is not None
            and card_plan not in {None, ProducePlanType.Common, profile.plan_type}
        )

        followup_playable: bool | None = None

        def has_followup_play() -> bool:
            nonlocal followup_playable
            if followup_playable is None:
                followup_playable = simulate_play(state, card).playable_count > 0
            return followup_playable

        grants_extra_turn = any(
            play_effect.produce_exam_effect is not None
            and play_effect.produce_exam_effect.effect_type == ProduceExamEffectType.ExamExtraTurn
            and evaluate_trigger_id(
                getattr(play_effect, '_produce_exam_trigger_id', ''), state
            ) != TriggerVerdict.INACTIVE
            for play_effect in card.play_effects
        )

        def deferred_value_multiplier() -> float:
            # These are option values, not confirmed output. A fixed reward
            # for entering preservation or growing a card assumes ample time
            # to turn the investment into points, which late lessons lack.
            # Same-turn follow-ups and extra turns preserve some option value;
            # shallow lookahead separately scores the actual visible follow-up.
            opportunities = max(0, turns - 1) + int(grants_extra_turn)
            if has_followup_play():
                opportunities += 1
            # Preservation needs a later stance exit and scoring play; growth
            # needs the affected card to be available and played again. Use a
            # conservative two-stage discount, not a promise of either event.
            return min(1.0, opportunities / 5.0) ** 2

        def setup_multiplier(
            kind: ProduceExamEffectType,
            *,
            needs_card_followup: bool = False,
        ) -> float:
            if (
                needs_card_followup
                and turns <= 1
                and not grants_extra_turn
                and not has_followup_play()
            ):
                return 0.0
            multiplier = profile.setup_multiplier(kind) if profile else 1.0
            if state.is_exam:
                # Exams need raw score. Keep native setup viable early, but stop
                # defensive or cross-plan setup from crowding out finishers.
                multiplier *= 0.7 if turns > 2 else 0.2
            if state.score_target_is_final and state.score_gap is not None:
                multiplier *= max(0.1, min(1.0, state.score_gap / 50.0))
            return multiplier * (0.35 if cross_plan else 1.0)

        output_multiplier = 1.0
        if state.buffs.parameter_buff_turns > 0:
            output_multiplier *= 1.5
        if state.buffs.parameter_buff_multiple_turns > 0:
            output_multiplier *= 1.1
        stance = 'normal' if state.anomaly.full_power_active else state.anomaly.stance.value
        if stance == 'aggressive':
            output_multiplier *= 2.0
        elif stance == 'aggressive_2':
            output_multiplier *= 2.5
        elif stance == 'preservation':
            output_multiplier *= 0.5
        elif stance == 'preservation_2':
            output_multiplier *= 0.25
        if state.anomaly.full_power_active:
            output_multiplier *= 3.0

        # Use the same integer payment as affordability and shallow search.
        green_cost, red_cost = play_cost(state, card)
        if state.genki_known and state.genki >= green_cost:
            green_weight = 1.0
        elif not state.hp_known:
            green_weight = 4.0
        else:
            green_weight = 2.0 if state.hp > 8 else 6.0
        score -= green_cost * green_weight
        red_weight = (
            1.5 if not state.hp_known
            else 2.0 if state.hp <= max(5, red_cost * 2)
            else 1.0
        )
        score -= red_cost * red_weight

        for play_effect in card.play_effects:
            effect = play_effect.produce_exam_effect
            if effect is None or effect.effect_type is None:
                continue
            trigger_verdict = evaluate_trigger_id(
                getattr(play_effect, '_produce_exam_trigger_id', ''),
                state,
            )
            if trigger_verdict == TriggerVerdict.INACTIVE:
                reasons.append('条件未满足')
                continue
            score_before_effect = score
            output_before_effect = projected_output
            kind = effect.effect_type
            value = float(effect.effect_value1 or 0)
            value2 = float(effect.effect_value2 or 0)
            count = max(1, effect.effect_count or 1)
            effect_turn = effect.effect_turn or 0
            duration = effect_turn if effect_turn > 0 else turns

            if kind in self._DIRECT:
                output = (value + state.anomaly.enthusiasm) * count * output_multiplier
                projected_output += output
                score += self._score_output(output, state) * late_weight
                reasons.append('即时参数')
            elif kind == ProduceExamEffectType.ExamLessonFullPowerPoint:
                output = (
                    value
                    + state.anomaly.cumulative_full_power_points * value2 / 1000.0
                    + state.anomaly.enthusiasm
                ) * count * output_multiplier
                projected_output += output
                score += self._score_output(output, state) * late_weight
                reasons.append('全力值依赖参数')
            elif kind in self._DEFENCE:
                if (
                    kind in {ProduceExamEffectType.ExamBlock, ProduceExamEffectType.ExamBlockFix}
                    and turns <= 1
                    and not grants_extra_turn
                    and not has_followup_play()
                    and not any(
                        other.produce_exam_effect is not None
                        and other.produce_exam_effect.effect_type in {
                            ProduceExamEffectType.ExamLessonDependBlock,
                            ProduceExamEffectType.ExamReviewDependExamBlock,
                        }
                        and evaluate_trigger_id(
                            getattr(other, '_produce_exam_trigger_id', ''), state
                        ) != TriggerVerdict.INACTIVE
                        for other in card.play_effects
                    )
                ):
                    # Genki cannot pay another card after the last play. HP
                    # recovery is different: it can carry into later lessons.
                    reasons.append('终局元气无后续利用')
                    continue
                if state.is_exam:
                    survival = (
                        1.1 if not state.hp_known
                        else 2.0 if state.hp <= 5 else 0.2
                    )
                else:
                    survival = (
                        1.625 if not state.hp_known
                        else 2.5 if state.hp <= 8 else 0.75
                    )
                score += value * survival
                reasons.append('生存')
            elif kind in self._SETUP:
                effective_value = value if value > 0 else max(1.0, float(duration))
                score += (
                    effective_value
                    * min(duration, early_weight)
                    * 0.8
                    * setup_multiplier(kind, needs_card_followup=True)
                )
                reasons.append('前置增益')
            elif kind == ProduceExamEffectType.ExamReview:
                score += value * min(turns, 5) * setup_multiplier(kind)
                reasons.append('好印象')
            elif kind in {
                ProduceExamEffectType.ExamReviewAdditive,
                ProduceExamEffectType.ExamReviewMultiple,
                ProduceExamEffectType.ExamReviewValueMultiple,
                ProduceExamEffectType.ExamLessonAddMultipleLessonBuff,
                ProduceExamEffectType.ExamParameterBuffMultiplePerTurn,
                ProduceExamEffectType.ExamFullPowerPointAdditive,
                ProduceExamEffectType.ExamFullPowerLessonMultipleAdditive,
            }:
                score += (value / 100.0) * min(turns, 5) * setup_multiplier(kind)
                reasons.append('体系强化')
            elif kind == ProduceExamEffectType.ExamMultipleLessonBuffLesson:
                output = (
                    value
                    + state.buffs.lesson_buff * value2 / 1000.0
                    + state.anomaly.enthusiasm
                ) * count * output_multiplier
                projected_output += output
                score += self._score_output(output, state) * late_weight
                reasons.append('集中依赖参数')
            elif kind == ProduceExamEffectType.ExamLessonDependParameterBuff:
                if state.buffs.parameter_buff_turns > 0:
                    output = value / 1000.0 * output_multiplier
                    projected_output += output
                    score += self._score_output(output, state) * late_weight
                reasons.append('好调依赖参数')
            elif kind == ProduceExamEffectType.ExamLessonDependExamCardPlayAggressive:
                output = (
                    state.buffs.aggressive
                    * value
                    / 1000.0
                    * output_multiplier
                )
                projected_output += output
                score += self._score_output(output, state) * late_weight
                reasons.append('干劲依赖参数')
            elif kind == ProduceExamEffectType.ExamLessonDependExamReview:
                output = (
                    state.buffs.review
                    * value
                    / 1000.0
                    * output_multiplier
                )
                projected_output += output
                score += self._score_output(output, state) * late_weight
                reasons.append('好印象依赖参数')
            elif kind == ProduceExamEffectType.ExamLessonDependBlock:
                output = state.genki * value / 1000.0 * output_multiplier
                projected_output += output
                score += self._score_output(output, state) * late_weight
                reasons.append('元气依赖参数')
            elif kind == ProduceExamEffectType.ExamBlockAddMultipleAggressive:
                score += state.buffs.aggressive * value / 1000.0
                reasons.append('干劲依赖元气')
            elif kind == ProduceExamEffectType.ExamMultipleEnthusiasticLesson:
                output = (
                    state.anomaly.enthusiasm * value / 1000.0
                ) * count * output_multiplier
                projected_output += output
                score += self._score_output(output, state) * late_weight
                reasons.append('热意依赖参数')
            elif kind in {
                ProduceExamEffectType.ExamCardDraw,
                ProduceExamEffectType.ExamHandGraveCountCardDraw,
            }:
                # Memory only contains cards already seen in hand. An empty
                # inference cannot establish that the actual deck is empty.
                if state.draw_pile_card_ids or state.draw_pile_empty is False:
                    draw_value = 18.0
                elif state.draw_pile_empty is True:
                    draw_value = 8.0
                else:
                    draw_value = 13.0
                # The hand is discarded at turn end. A draw with no plays
                # remaining has little value, unless this same card grants a
                # play. Reuse the transition from shallow lookahead so that
                # conditional extra plays are handled consistently.
                if not has_followup_play():
                    draw_value *= 0.1
                    reasons.append('过牌后无出牌次数')
                # Fixed draw effects store the number of cards in value1;
                # effect_count is zero for the game's draw-1..draw-5 rows.
                # HandGraveCountCardDraw replaces the remaining hand. Its
                # actual draw count depends on the visible hand size after
                # this card leaves it; unknown size keeps a neutral estimate.
                draw_count = (
                    max(1, int(value))
                    if kind == ProduceExamEffectType.ExamCardDraw
                    else (
                        max(0, state.hand_size - 1)
                        if state.hand_size is not None
                        else 1
                    )
                )
                score += draw_count * draw_value
                reasons.append('过牌')
            elif kind == ProduceExamEffectType.ExamPlayableValueAdd:
                # The shallow planner separately adds the best actual follow-up,
                # so this is only the intrinsic flexibility value.
                score += max(1.0, value) * 12.0
                reasons.append('追加出牌')
            elif kind == ProduceExamEffectType.ExamExtraTurn:
                score += 60.0
                reasons.append('追加回合')
            elif kind == ProduceExamEffectType.ExamPreservation:
                base = 14.0 if state.anomaly.stance.value == 'normal' else 4.0
                score += (
                    base * setup_multiplier(kind, needs_card_followup=True)
                    * deferred_value_multiplier()
                )
                reasons.append('温存姿态')
            elif kind == ProduceExamEffectType.ExamOverPreservation:
                base = 24.0 if state.anomaly.stance.value.startswith('preservation') else 5.0
                score += (
                    base * setup_multiplier(kind, needs_card_followup=True)
                    * deferred_value_multiplier()
                )
                reasons.append('温存联动')
            elif kind == ProduceExamEffectType.ExamFullPowerPoint:
                # Reaching the threshold is one transition, not a new full-
                # power activation for every subsequent point card. Pending
                # full power may already be known even if its number OCR is not.
                activates = (
                    not state.anomaly.full_power_pending
                    and state.anomaly.full_power_points < 10
                    <= state.anomaly.full_power_points + value
                )
                can_use_next_turn = turns > 1
                point_value = (
                    6.0 if activates and can_use_next_turn else (1.5 + 0.3 * turns)
                )
                if not can_use_next_turn:
                    point_value = 0.15
                score += value * point_value * setup_multiplier(kind)
                if activates and can_use_next_turn:
                    score += 35.0
                elif state.anomaly.full_power_pending or state.anomaly.full_power_points >= 10:
                    reasons.append('全力已待触发，不重复计启动收益')
                reasons.append('全力蓄积')
            elif kind == ProduceExamEffectType.ExamFullPower:
                score += 55.0 if not state.anomaly.full_power_active else 8.0
                reasons.append('全力启动')
            elif kind == ProduceExamEffectType.ExamEnthusiasticAdditive:
                score += value * (1.5 + 0.15 * early_weight) * setup_multiplier(kind)
                reasons.append('热意')
            elif kind == ProduceExamEffectType.ExamEnthusiasticMultiple:
                score += (
                    state.anomaly.enthusiasm
                    * max(1.0, value / 100.0)
                    * setup_multiplier(kind)
                )
                reasons.append('热意倍率')
            elif kind == ProduceExamEffectType.ExamStatusEnchant:
                enchant_prior = self._status_enchant_prior(effect, state)
                score += enchant_prior * (0.35 if cross_plan else 1.0)
                reasons.append(
                    '状态附魔效果链' if enchant_prior != 0 else '状态附魔本次未计分'
                )
            elif kind == ProduceExamEffectType.ExamAddGrowEffect:
                score += (
                    10.0 * (0.35 if cross_plan else 1.0)
                    * deferred_value_multiplier()
                )
                reasons.append('成长效果')
            elif kind == ProduceExamEffectType.ExamEffectTimer:
                score += 6.0 * (0.35 if cross_plan else 1.0)
                reasons.append('延迟触发')
            elif kind in {
                ProduceExamEffectType.ExamCardMove,
                ProduceExamEffectType.ExamCardCreateId,
                ProduceExamEffectType.ExamCardCreateSearch,
                ProduceExamEffectType.ExamForcePlayCardSearch,
            }:
                score += 8.0
                reasons.append('牌区操作')

            official_prior = self._official_effect_prior(kind, state)
            if kind in {
                ProduceExamEffectType.ExamPreservation,
                ProduceExamEffectType.ExamOverPreservation,
                ProduceExamEffectType.ExamAddGrowEffect,
            }:
                # Do not let a generic positive prior undo the horizon limit.
                # Negative side-effect priors remain conservative penalties.
                if official_prior > 0:
                    official_prior *= deferred_value_multiplier()
                if deferred_value_multiplier() < 1:
                    reasons.append('后续兑现机会有限')
            if official_prior != 0:
                score += official_prior
                reasons.append(
                    '官方体系权重' if official_prior > 0 else '官方副作用权重'
                )

            if trigger_verdict == TriggerVerdict.UNKNOWN:
                # Keep some option value for conditions the HUD cannot expose,
                # but never score them as though they certainly trigger.
                score = score_before_effect + (score - score_before_effect) * 0.35
                projected_output = output_before_effect + (
                    projected_output - output_before_effect
                ) * 0.35
                reasons.append('条件未确认')
            else:
                confirmed_output += max(0.0, projected_output - output_before_effect)

        if cross_plan:
            reasons.append('活动跨流派牌')
        if (
            state.score_target_is_final
            and state.score_gap is not None
            and projected_output > 0
        ):
            # All direct effects on a card share late_weight. Cap their total,
            # rather than each effect separately, once PERFECT is the target.
            score -= max(0.0, projected_output - max(0, state.score_gap)) * late_weight
            if (
                state.score_gap > 0
                and conservative_target_output(confirmed_output) >= state.score_gap
            ):
                score += 35.0
                reasons.append('足量达标')
        elif (
            state.score_target_is_final is False
            and turns <= 1
            and state.score_gap is not None
            and state.score_gap > 0
            and conservative_target_output(confirmed_output) >= state.score_gap
        ):
            # CLEAR is normally an intermediate milestone, but it becomes a
            # hard deadline on the final lesson turn. Do not cap overkill:
            # additional output may still reach PERFECT.
            score += 35.0
            reasons.append('终局CLEAR达标')
        detail = '、'.join(dict.fromkeys(reasons)) or '费用与卡牌基础值'
        return CardEvaluation(
            card._id, score, EvaluationConfidence.LOW, detail,
            projected_output=projected_output,
            confirmed_output=confirmed_output,
        )

    @classmethod
    def _official_effect_prior(
        cls,
        kind: ProduceExamEffectType,
        state: BattleState,
        *,
        status_enchant: bool = False,
    ) -> float:
        """Convert large official coefficients into a bounded planner prior."""
        if state.archetype is None:
            return 0.0
        suffix = cls._OFFICIAL_EVALUATION_TYPES.get(kind)
        if suffix is None and status_enchant:
            suffix = cls._STATUS_ENCHANT_EVALUATION_TYPES.get(kind)
        if suffix is None:
            return 0.0
        row = get_auto_effect_evaluation(
            state.archetype.value,
            state.remaining_turns,
            f'ProduceExamAutoEvaluationType_{suffix}',
        )
        if row is None or abs(row.evaluation) <= 1:
            return 0.0
        # Coefficients span 1..~150k. A logarithmic, capped prior preserves
        # their ordering without overwhelming actual output and survival math.
        magnitude = min(12.0, math.log10(abs(row.evaluation)) * 2.5)
        return math.copysign(magnitude, row.evaluation)

    @classmethod
    def _status_enchant_prior(
        cls,
        effect: ProduceExamEffect,
        state: BattleState,
    ) -> float:
        enchant_id = getattr(effect, '_produce_exam_status_enchant_id', '')
        enchant = get_exam_status_enchant(enchant_id)
        if enchant is None or not enchant.effects:
            return 0.0
        condition = load_exam_trigger(enchant.trigger_id)
        phase_types = condition.phase_types if condition is not None else ()
        if 'ProduceExamPhaseType_StartPlay' in phase_types:
            return 0.0
        if (
            'ProduceExamPhaseType_ExamStartTurn' in phase_types
            and state.remaining_turns <= 1
        ):
            return 0.0
        nested = sum(
            cls._official_effect_prior(
                nested_effect.effect_type,
                state,
                status_enchant=True,
            )
            for nested_effect in enchant.effects
            if nested_effect.effect_type is not None
        )
        nested += 0.5 * sum(
            cls._official_effect_prior(
                nested_effect.effect_type,
                state,
                status_enchant=True,
            )
            for nested_effect in enchant.chained_effects
            if nested_effect.effect_type is not None
        )
        if nested == 0:
            return 0.0
        trigger = get_auto_trigger_evaluation(enchant.trigger_id)
        expected_factor = (
            max(0.0, min(2.0, trigger.coefficient_permil / 1000.0))
            if trigger is not None
            else 0.35
        )
        verdict = evaluate_trigger_id(enchant.trigger_id, state)
        if verdict == TriggerVerdict.ACTIVE:
            expected_factor = max(1.0, expected_factor)
        elif verdict == TriggerVerdict.INACTIVE:
            expected_factor *= 0.35
        else:
            expected_factor *= 0.6
        duration = effect.effect_turn or 0
        opportunities = min(
            state.remaining_turns - int(
                'ProduceExamPhaseType_ExamStartTurn' in phase_types
            ),
            duration if duration > 0 else state.remaining_turns,
        )
        count_limit = effect.effect_count or 0
        if count_limit > 0:
            opportunities = min(opportunities, count_limit)
        expected_factor *= min(1.0, max(0, opportunities) / 3.0)
        return max(-12.0, min(12.0, nested * expected_factor))

    @staticmethod
    def _score_output(output: float, state: BattleState) -> float:
        """CLEAR is only an intermediate lesson target; keep output beyond it."""
        if state.is_exam:
            return output * 1.6
        return output


class CompositeEvaluator:
    """Use official data as a bounded prior, never as an absolute command."""

    def __init__(
        self,
        official: CardEvaluator | None = None,
        heuristic: CardEvaluator | None = None,
    ) -> None:
        self._official = official or OfficialAutoEvaluator()
        self._heuristic = heuristic or EffectHeuristicEvaluator()

    def evaluate(self, card: SkillCard, state: BattleState) -> CardEvaluation:
        official = self._official.evaluate(card, state)
        heuristic = self._heuristic.evaluate(card, state)
        if heuristic is None:
            heuristic = CardEvaluation(
                card._id,
                0.0,
                EvaluationConfidence.LOW,
                '无可模拟效果',
            )
        if official is None:
            return heuristic
        profile = profile_for(state.archetype)
        try:
            card_plan = ProducePlanType(card.plan_type) if card.plan_type else None
        except ValueError:
            card_plan = None
        cross_plan = bool(
            profile
            and card_plan not in {None, ProducePlanType.Common, profile.plan_type}
        )
        if official.score > 0:
            official_bias = 12.0 if cross_plan else 45.0
        elif official.score < 0:
            official_bias = -60.0
        else:
            official_bias = 0.0
        terminal_without_output = (
            state.remaining_turns <= 1
            and heuristic.confirmed_output <= 0
            and simulate_play(state, card).playable_count <= 0
            and not any(
                effect.produce_exam_effect is not None
                and effect.produce_exam_effect.effect_type
                == ProduceExamEffectType.ExamExtraTurn
                and evaluate_trigger_id(
                    getattr(effect, '_produce_exam_trigger_id', ''), state
                ) != TriggerVerdict.INACTIVE
                for effect in card.play_effects
            )
        )
        if terminal_without_output:
            # The official table is a general prior, not a reason to prefer
            # a non-scoring setup card when the lesson ends after this play.
            official_bias = min(0.0, official_bias)
        return CardEvaluation(
            card_id=card._id,
            score=official_bias + heuristic.score,
            confidence=official.confidence,
            reason=(
                f'{official.reason}（先验 {official_bias:+.0f}）；'
                f'状态评分：{heuristic.reason}'
                + ('；终局无确定得分' if terminal_without_output else '')
            ),
            projected_output=heuristic.projected_output,
            confirmed_output=heuristic.confirmed_output,
        )
