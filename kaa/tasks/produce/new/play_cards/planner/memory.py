"""A small, screenshot-tolerant memory for one lesson or exam."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

from kotonebot import logging

from kaa.db.constants import ProduceExamEffectType
from kaa.db.skill_card import SkillCard

from .state import AnomalyStance, AnomalyState, BattleState, ExamBuffState
from .trigger import TriggerVerdict, evaluate_trigger_id

FULL_POWER_THRESHOLD = 10
logger = logging.getLogger(__name__)
PRESERVATION_STANCES = {
    AnomalyStance.PRESERVATION,
    AnomalyStance.PRESERVATION_2,
}
AGGRESSIVE_STANCES = {
    AnomalyStance.AGGRESSIVE,
    AnomalyStance.AGGRESSIVE_2,
}


@dataclass
class BattleMemory:
    """Track card zones and the anomaly state that can be inferred from our plays.

    Screenshots expose the hand, but not the complete draw/discard piles.  The
    memory therefore never invents unseen cards: it records the greatest known
    multiplicity of every observed card and reconciles zones on every decision.
    """

    known_cards: Counter[str] = field(default_factory=Counter)
    discard: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    previous_hand: tuple[str, ...] = ()
    previous_remaining_turns: int | None = None
    pending_turn_correction: int | None = None
    turn_baseline_known: bool = False
    plays_this_turn: int = 0
    anomaly: AnomalyState = field(default_factory=AnomalyState)
    buffs: ExamBuffState = field(default_factory=ExamBuffState)
    effect_misses: Counter[str] = field(default_factory=Counter)

    def reconcile_remaining_turns(self, observed: int | None) -> int:
        """Debounce OCR jumps without permanently locking a stale turn count.

        A normal reading is unchanged or one turn lower. Larger discrepancies
        need two consistent readings, including recovery from the initial
        fallback value when the HUD could not be read.
        """
        previous = self.previous_remaining_turns
        if observed is not None and not 1 <= observed <= 30:
            observed = None
        if previous is None:
            return observed if observed is not None and observed > 0 else 7
        if observed is None:
            self.pending_turn_correction = None
            return previous
        if previous - 1 <= observed <= previous:
            self.pending_turn_correction = None
            return observed
        pending = self.pending_turn_correction
        if (
            pending is not None
            and abs(observed - pending) <= 1
            and (observed > previous) == (pending > previous)
        ):
            self.pending_turn_correction = None
            logger.warning(
                'Planner corrected remaining turns after repeated HUD readings: %d -> %d',
                previous,
                observed,
            )
            return observed
        self.pending_turn_correction = observed
        return previous if observed > previous else previous - 1

    def observe(self, remaining_turns: int, hand: list[SkillCard]) -> None:
        ids = tuple(card._id for card in hand)
        hand_counts = Counter(ids)
        for card_id, count in hand_counts.items():
            self.known_cards[card_id] = max(self.known_cards[card_id], count)

        if (
            self.previous_remaining_turns is not None
            and remaining_turns < self.previous_remaining_turns
        ):
            # Cards left in hand are normally sent to the grave at turn end.
            self.discard.extend(self.previous_hand)
            elapsed_turns = self.previous_remaining_turns - remaining_turns
            # OCR recovery and skipped/unplayable turns can jump by more than
            # one. Advance every inferred turn, but discard the last visible
            # hand only once: unseen intervening hands must not be invented.
            for _ in range(elapsed_turns):
                self.anomaly = self._advance_anomaly_turn(self.anomaly)
            self.plays_this_turn = 0
            self.buffs = replace(
                self.buffs,
                parameter_buff_turns=max(0, self.buffs.parameter_buff_turns - elapsed_turns),
                parameter_buff_multiple_turns=max(
                    0, self.buffs.parameter_buff_multiple_turns - elapsed_turns
                ),
            )

        self.previous_hand = ids
        self.previous_remaining_turns = remaining_turns

    def reconcile_effect_values(self, observed: dict[str, int | None]) -> None:
        """Reconcile inferred buffs with debounced current HUD readings."""
        points = observed.get('full_power_point')
        if points is not None and points >= 0:
            # Current points can be corrected without inventing cumulative
            # gains, an activation or an extra play that has not happened yet.
            self.anomaly = replace(
                self.anomaly,
                full_power_points=points,
                full_power_pending=points >= FULL_POWER_THRESHOLD,
            )
        changes: dict[str, int] = {}
        fields = {
            'lesson_buff': 'lesson_buff',
            'parameter_buff': 'parameter_buff_turns',
            'review': 'review',
        }
        for effect, field_name in fields.items():
            if effect in observed:
                self.effect_misses[effect] = 0
                value = observed[effect]
                if value is not None:
                    changes[field_name] = value
                elif getattr(self.buffs, field_name) == 0:
                    # A recognized icon establishes presence even when its
                    # numeric label has not been calibrated for OCR.
                    changes[field_name] = 1
                continue
            self.effect_misses[effect] += 1
            # A single miss can be animation or template noise. Two decisions
            # without the icon are strong evidence that the effect expired.
            if self.effect_misses[effect] >= 2:
                changes[field_name] = 0
        if changes:
            self.buffs = replace(self.buffs, **changes)
        if ('concentration' in observed and 'full_power' not in observed
                and not self.anomaly.full_power_active):
            # A visible strong-stance glyph corrects missed card/item changes.
            # Presence does not distinguish strong from super-strong; retain
            # a known higher level, otherwise use the conservative base level.
            # This is reconciliation, not another played stance transition:
            # do not invent enthusiasm or increment transition counters.
            if self.anomaly.stance not in AGGRESSIVE_STANCES:
                self.anomaly = replace(self.anomaly, stance=AnomalyStance.AGGRESSIVE)
            # Absence alone is deliberately not used to reset the stance:
            # the uncalibrated higher-level glyph or animation can hide it.
        if 'full_power' in observed:
            self.effect_misses['full_power'] = 0
            # Full power is itself the guiding stance: no preservation/strong
            # stance may coexist, even when its activation was missed.
            self.anomaly = replace(
                self.anomaly, full_power_active=True, stance=AnomalyStance.NORMAL,
            )
        elif self.anomaly.full_power_turns > 0:
            # A play/turn transition already established this full-power turn.
            # Missing the HUD glyph must not cancel it halfway through the turn.
            self.effect_misses['full_power'] = 0
        else:
            self.effect_misses['full_power'] += 1
            if self.effect_misses['full_power'] >= 2:
                self.anomaly = replace(
                    self.anomaly,
                    full_power_active=False,
                    full_power_turns=0,
                )

    @staticmethod
    def _advance_anomaly_turn(anomaly: AnomalyState) -> AnomalyState:
        """Apply anomaly effects that happen at the start of the next turn."""
        points = anomaly.full_power_points
        activates = anomaly.full_power_pending or points >= FULL_POWER_THRESHOLD
        enthusiasm = 0
        stance = anomaly.stance
        if activates:
            preservation_exit = stance in PRESERVATION_STANCES
            if stance is AnomalyStance.PRESERVATION:
                enthusiasm = 5
            elif stance is AnomalyStance.PRESERVATION_2:
                enthusiasm = 8
            points = max(0, points - FULL_POWER_THRESHOLD)
            stance = AnomalyStance.NORMAL
        return replace(
            anomaly,
            stance=stance,
            stance_locked=False,
            full_power_points=points,
            full_power_pending=points >= FULL_POWER_THRESHOLD,
            full_power_active=activates,
            full_power_turns=1 if activates else 0,
            full_power_change_count=(
                anomaly.full_power_change_count + int(activates)
            ),
            enthusiasm=enthusiasm,
            extra_play_count=(1 + int(preservation_exit)) if activates else 0,
        )

    def record_play(self, card: SkillCard, state: BattleState | None = None) -> None:
        try:
            hand = list(self.previous_hand)
            hand.remove(card._id)
            self.previous_hand = tuple(hand)
        except ValueError:
            pass

        if getattr(card, 'play_move_position_type', None) == 'ProduceCardMovePositionType_Grave':
            self.discard.append(card._id)
        else:
            self.removed.append(card._id)
        self._apply_anomaly_effects(card, state)
        self.plays_this_turn += 1

    def inferred_draw_pile(self, hand: list[SkillCard]) -> tuple[str, ...]:
        remaining = self.known_cards.copy()
        remaining.subtract(card._id for card in hand)
        remaining.subtract(self.discard)
        remaining.subtract(self.removed)
        return tuple(
            card_id
            for card_id, count in remaining.items()
            for _ in range(max(0, count))
        )

    def _apply_anomaly_effects(
        self,
        card: SkillCard,
        state: BattleState | None = None,
    ) -> None:
        anomaly = self.anomaly
        buffs = self.buffs
        cost_type = getattr(card, 'cost_type', None)
        cost_value = max(0, getattr(card, 'cost_value', 0) or 0)
        if cost_type == 'ExamCostType_ExamFullPowerPoint':
            anomaly = replace(
                anomaly,
                full_power_points=max(0, anomaly.full_power_points - cost_value),
                full_power_pending=(
                    anomaly.full_power_points - cost_value >= FULL_POWER_THRESHOLD
                ),
            )
        elif cost_type == 'ExamCostType_ExamLessonBuff':
            buffs = replace(buffs, lesson_buff=max(0, buffs.lesson_buff - cost_value))
        elif cost_type == 'ExamCostType_ExamReview':
            buffs = replace(buffs, review=max(0, buffs.review - cost_value))
        elif cost_type == 'ExamCostType_ExamParameterBuff':
            buffs = replace(
                buffs, parameter_buff_turns=max(0, buffs.parameter_buff_turns - cost_value)
            )
        elif cost_type == 'ExamCostType_ExamParameterBuffMultiplePerTurn':
            buffs = replace(
                buffs, parameter_buff_multiple_turns=max(
                    0, buffs.parameter_buff_multiple_turns - cost_value
                ),
            )
        elif cost_type == 'ExamCostType_ExamCardPlayAggressive':
            buffs = replace(buffs, aggressive=max(0, buffs.aggressive - cost_value))

        for play_effect in getattr(card, 'play_effects', ()):
            effect = play_effect.produce_exam_effect
            if effect is None or effect.effect_type is None:
                continue
            if (
                state is not None
                and evaluate_trigger_id(
                    getattr(play_effect, '_produce_exam_trigger_id', ''),
                    state,
                ) != TriggerVerdict.ACTIVE
            ):
                continue
            value = effect.effect_value1 or 0
            match effect.effect_type:
                case ProduceExamEffectType.ExamPreservation:
                    stance = (
                        AnomalyStance.PRESERVATION_2
                        if value >= 2
                        or anomaly.stance in PRESERVATION_STANCES
                        else AnomalyStance.PRESERVATION
                    )
                    if not anomaly.stance_locked and not anomaly.full_power_active:
                        anomaly = self._change_stance(anomaly, stance)
                case ProduceExamEffectType.ExamConcentration:
                    stance = (
                        AnomalyStance.AGGRESSIVE_2
                        if value >= 2
                        or anomaly.stance in AGGRESSIVE_STANCES
                        else AnomalyStance.AGGRESSIVE
                    )
                    if not anomaly.stance_locked and not anomaly.full_power_active:
                        anomaly = self._change_stance(anomaly, stance)
                case ProduceExamEffectType.ExamStanceReset:
                    if not anomaly.stance_locked and not anomaly.full_power_active:
                        anomaly = self._change_stance(anomaly, AnomalyStance.NORMAL)
                case ProduceExamEffectType.StanceLock:
                    anomaly = replace(anomaly, stance_locked=True)
                case ProduceExamEffectType.ExamOverPreservation:
                    anomaly = replace(anomaly, over_preservation=True)
                case ProduceExamEffectType.ExamFullPowerPoint:
                    points = max(0, anomaly.full_power_points + value)
                    anomaly = replace(
                        anomaly,
                        full_power_points=points,
                        cumulative_full_power_points=(
                            anomaly.cumulative_full_power_points + max(0, value)
                        ),
                        full_power_pending=points >= FULL_POWER_THRESHOLD,
                    )
                case ProduceExamEffectType.ExamFullPowerPointReduce:
                    points = max(0, anomaly.full_power_points - value)
                    anomaly = replace(
                        anomaly,
                        full_power_points=points,
                        full_power_pending=points >= FULL_POWER_THRESHOLD,
                    )
                case ProduceExamEffectType.ExamFullPower:
                    newly_active = not anomaly.full_power_active
                    anomaly = replace(
                        anomaly,
                        full_power_active=True,
                        stance=AnomalyStance.NORMAL,
                        full_power_turns=max(anomaly.full_power_turns, effect.effect_turn or 1),
                        full_power_change_count=(
                            anomaly.full_power_change_count + int(newly_active)
                        ),
                        extra_play_count=(
                            anomaly.extra_play_count
                            + int(newly_active)
                        ),
                    )
                case ProduceExamEffectType.ExamEnthusiasticAdditive:
                    anomaly = replace(
                        anomaly,
                        enthusiasm=max(0, anomaly.enthusiasm + value),
                    )
                case ProduceExamEffectType.ExamPlayableValueAdd:
                    anomaly = replace(
                        anomaly,
                        extra_play_count=anomaly.extra_play_count + max(1, value),
                    )
                case ProduceExamEffectType.ExamLessonBuff:
                    buffs = replace(buffs, lesson_buff=max(0, buffs.lesson_buff + value))
                case ProduceExamEffectType.ExamParameterBuff:
                    buffs = replace(
                        buffs,
                        parameter_buff_turns=max(
                            buffs.parameter_buff_turns, effect.effect_turn or value
                        ),
                    )
                case ProduceExamEffectType.ExamParameterBuffMultiplePerTurn:
                    buffs = replace(
                        buffs,
                        parameter_buff_multiple_turns=max(
                            buffs.parameter_buff_multiple_turns,
                            effect.effect_turn or value,
                        ),
                    )
                case ProduceExamEffectType.ExamReview:
                    buffs = replace(buffs, review=max(0, buffs.review + value))
                case ProduceExamEffectType.ExamCardPlayAggressive:
                    buffs = replace(buffs, aggressive=max(0, buffs.aggressive + value))
                case _:
                    pass
        self.anomaly = anomaly
        self.buffs = buffs

    @staticmethod
    def _change_stance(
        anomaly: AnomalyState, stance: AnomalyStance
    ) -> AnomalyState:
        """Change stance and grant enthusiasm when preservation is left."""
        enthusiasm = anomaly.enthusiasm
        left_preservation = (
            anomaly.stance in PRESERVATION_STANCES
            and stance not in PRESERVATION_STANCES
        )
        if left_preservation:
            enthusiasm += (
                8 if anomaly.stance is AnomalyStance.PRESERVATION_2 else 5
            )
        return replace(
            anomaly,
            stance=stance,
            stance_change_count=anomaly.stance_change_count + 1,
            preservation_change_count=(
                anomaly.preservation_change_count
                + int(stance in PRESERVATION_STANCES)
            ),
            concentration_change_count=(
                anomaly.concentration_change_count
                + int(stance in AGGRESSIVE_STANCES)
            ),
            enthusiasm=enthusiasm,
            extra_play_count=anomaly.extra_play_count + int(left_preservation),
        )
