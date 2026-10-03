"""独立于旧专家规则的规划器策略入口。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

from kotonebot import logging
from typing_extensions import override

from kaa.kaa_context import produce_session
from kaa.db.constants import ProduceExamEffectType, ProducePlanType

from ..strategy import AbstractBattleStrategy
from .evaluator import (
    CardEvaluation,
    CardEvaluator,
    CompositeEvaluator,
    conservative_target_output,
)
from .feedback import PendingPlay, PlayFeedback, PlayFeedbackStatus, evaluate_play_feedback
from .decision_trace import DecisionTrace
from .hud_effects import observe_effect_icons
from .memory import BattleMemory
from .play_evidence import PlayFeedbackEvidence
from .profiles import profile_for
from .simulator import can_afford_play, play_cost, simulate_play
from .state import BattleState

if TYPE_CHECKING:
    from kaa.db.skill_card import SkillCard
    from ..page import LessonBattleContext
    from ..ui import CardGameObject

logger = logging.getLogger(__name__)


class PlannerStrategy(AbstractBattleStrategy):
    """数据驱动的第三套出牌策略。

    结合可观察 HUD 状态、局内记忆与浅层搜索进行评价；无法评价时返回
    ``False``，由现有战斗循环降级到黄色推荐卡。遗物触发尚未纳入评价。
    """

    def __init__(
        self,
        evaluator: CardEvaluator | None = None,
        *,
        is_exam: bool = False,
    ) -> None:
        self._evaluator = evaluator or CompositeEvaluator()
        self._memory = BattleMemory()
        self._is_exam = is_exam
        self._last_move_state: BattleState | None = None
        self._pending_play: PendingPlay | None = None
        self._memory_before_pending_play: BattleMemory | None = None
        self._play_evidence = PlayFeedbackEvidence.from_environment()
        self._decision_trace = DecisionTrace.from_environment()

    @override
    def on_battle_end(self) -> None:
        if self._pending_play is None:
            return
        feedback = PlayFeedback(
            PlayFeedbackStatus.UNCONFIRMED, 'battle_ended_without_observation',
        )
        if self._decision_trace is not None:
            self._decision_trace.feedback(feedback)
        if self._play_evidence is not None:
            self._play_evidence.feedback(feedback.evidence, None)
        self._pending_play = None
        self._memory_before_pending_play = None
        logger.info('Planner final request has no post-play observation at battle end.')

    @override
    def on_action(self, ctx: 'LessonBattleContext') -> bool:
        observed_hands = ctx.fetch_hands()
        hands = [
            card
            for card in observed_hands
            if card is not None and card.available and card.card is not None
        ]
        if not hands:
            logger.info('Planner found no recognized playable cards.')
            return False

        state = self._build_observed_state(ctx, observed_hands)
        if self._pending_play is not None:
            pending = self._pending_play
            feedback = evaluate_play_feedback(pending, state)
            if self._decision_trace is not None:
                self._decision_trace.feedback(feedback, state)
            if self._play_evidence is not None:
                try:
                    feedback_screen = ctx.fetch_screen()
                except Exception:
                    logger.warning('Could not capture feedback evidence frame.', exc_info=True)
                    feedback_screen = None
                self._play_evidence.feedback(feedback.evidence, feedback_screen)
            logger.info(
                'Planner play feedback: card=%s status=%s evidence=%s '
                'predicted=%.1f confirmed_estimate=%.1f gap_reduction=%s',
                pending.card_id, feedback.status.value, feedback.evidence,
                pending.projected_output, pending.confirmed_output,
                feedback.observed_gap_reduction,
            )
            # Unconfirmed is not failure. Remove only speculative effects and
            # reapply current observations; invisible successful plays remain
            # unknown instead of becoming confirmed buffs on every retry.
            if (
                feedback.status is PlayFeedbackStatus.UNCONFIRMED
                and self._memory_before_pending_play is not None
            ):
                logger.warning(
                    'Planner discarded speculative effects; play remains unconfirmed: %s',
                    pending.card_id,
                )
                self._memory = self._memory_before_pending_play
                self._last_move_state = pending.before
                state = self._build_observed_state(ctx, observed_hands)
            self._pending_play = None
            self._memory_before_pending_play = None
        self._last_move_state = state
        candidates: list[tuple[CardGameObject, CardEvaluation]] = [
            (card, evaluation)
            for card in hands
            if card.card is not None
            for evaluation in [self._evaluator.evaluate(card.card, state)]
            if evaluation is not None
        ]
        if not candidates:
            logger.info('Planner has no evaluation data; falling back to game recommendation.')
            return False

        candidates = self._add_shallow_lookahead(candidates, state)
        # Keep all cards in lookahead: a recovery card may make a currently
        # unaffordable finisher affordable. Only the first play is filtered.
        candidates = [
            (card, evaluation) for card, evaluation in candidates
            if card.card is not None and can_afford_play(state, card.card) is not False
        ]
        if not candidates:
            logger.info('Planner found no cards with sufficient observed resources; falling back.')
            return False
        candidates = self._protect_lesson_scoring_budget(candidates, state)
        candidates = self._protect_exam_finish(candidates, state)
        candidates = self._prefer_exam_catch_up_output(candidates, state)
        candidates = self._prefer_lesson_finish_progress(candidates, state)
        logger.debug(
            'Planner candidates (hp=%s, genki=%s, plays=%s, gap=%s): %s',
            state.hp if state.hp_known else 'unknown',
            state.genki if state.genki_known else 'unknown',
            state.playable_count,
            state.score_gap,
            [
                f'{card.card.name if card.card else evaluation.card_id}'
                f'={evaluation.score:.1f}/output {evaluation.projected_output:.1f}'
                f'/confirmed {evaluation.confirmed_output:.1f}'
                for card, evaluation in candidates
            ],
        )

        traced_candidates = candidates
        if (
            not state.is_exam
            and state.score_target_is_final is False
            and state.score_gap is not None
            and state.score_gap > 0
            and state.remaining_turns <= 4
        ):
            guaranteed_clear = [
                candidate for candidate in candidates
                if (
                    conservative_target_output(candidate[1].confirmed_output)
                    >= state.score_gap
                )
            ]
            if guaranteed_clear:
                logger.info(
                    'Planner prioritizes guaranteed CLEAR with %d turns left and %d play remaining.',
                    state.remaining_turns, state.playable_count,
                )
                candidates = guaranteed_clear

        best_card, best = max(candidates, key=lambda item: item[1].score)
        if self._decision_trace is not None:
            self._decision_trace.decision(state, traced_candidates, best)
        logger.info(
            'Planner selected %s: score=%.1f confidence=%s reason=%s',
            best_card.card.name if best_card.card else best.card_id,
            best.score,
            best.confidence.value,
            best.reason,
        )
        memory_before_play = deepcopy(self._memory)
        if self._play_evidence is not None:
            self._play_evidence.before_commit(best.card_id, best_card._screenshot)
        ctx.commit(best_card)
        if self._play_evidence is not None:
            self._play_evidence.after_commit()
        if best_card.card is not None:
            self._memory_before_pending_play = memory_before_play
            self._pending_play = PendingPlay(
                best_card.card._id,
                state,
                best.projected_output,
                best.confirmed_output,
                requires_resource_change=any(play_cost(state, best_card.card)),
            )
            self._memory.record_play(best_card.card, state)
            self._last_move_state = simulate_play(state, best_card.card)
        return True

    @override
    def rank_move_cards(self, cards: list['SkillCard | None']) -> list[int] | None:
        """Prefer cards valuable when hold returns them to the hand.

        Full-power decks normally hold payoff cards for the full-power turn,
        even when the next turn has not yet reached ten full-power points.
        A score gap visible before the move is deliberately not carried over:
        that target changes as cards score and would cap future payoff cards.
        """
        state = self._last_move_state
        if state is None:
            return None
        if self._pending_play is not None:
            self._pending_play = replace(self._pending_play, move_dialog_seen=True)
        anomaly = BattleMemory._advance_anomaly_turn(self._memory.anomaly)
        if state.archetype is ProduceExamEffectType.ExamFullPower:
            anomaly = replace(
                anomaly,
                full_power_active=True,
                full_power_turns=max(1, anomaly.full_power_turns),
            )
        future = replace(
            state,
            remaining_turns=max(1, state.remaining_turns - 1),
            score_gap=None,
            score_target_is_final=None,
            playable_count=2 if anomaly.full_power_active else 1,
            hand_card_ids=(),
            hand_size=None,
            anomaly=anomaly,
            buffs=self._memory.buffs,
        )
        scores: list[float | None] = []
        for card in cards:
            evaluation = self._evaluator.evaluate(card, future) if card is not None else None
            scores.append(
                0.5 * evaluation.score + 0.7 * evaluation.projected_output
                - (1000.0 if card is not None and can_afford_play(future, card) is False else 0.0)
                if evaluation is not None else None
            )
        if all(score is None for score in scores):
            return None
        def score_at(index: int) -> float:
            score = scores[index]
            return score if score is not None else float('-inf')
        ranked = sorted(
            range(len(cards)),
            key=score_at,
            reverse=True,
        )
        choices = []
        for index in ranked:
            card, score = cards[index], scores[index]
            choices.append(
                f'{card.name if card is not None else "unknown"}={score:.1f}'
                if score is not None else 'unknown'
            )
        logger.info('Planner hold choices: %s', choices)
        return ranked

    def _protect_lesson_scoring_budget(
        self,
        candidates: list[tuple['CardGameObject', CardEvaluation]],
        state: BattleState,
    ) -> list[tuple['CardGameObject', CardEvaluation]]:
        """Avoid spending the HP needed for later scoring on a setup-only turn.

        This is a local affordability check, not a claim that the same card
        will be drawn twice. It only applies when CLEAR needs substantially
        more output than one visible scorer can provide.
        """
        if (
            state.is_exam
            or state.score_target_is_final is not False
            or state.score_gap is None
            or state.remaining_turns <= 2
            or not state.hp_known
            or not state.genki_known
        ):
            return candidates

        def supports_two_scoring_plays(
            base: BattleState, scorer: 'CardGameObject'
        ) -> bool:
            return (
                scorer.card is not None
                and can_afford_play(base, scorer.card) is True
                and can_afford_play(
                    simulate_play(base, scorer.card), scorer.card
                ) is True
            )

        scorers = [
            (card, evaluation)
            for card, evaluation in candidates
            if card.card is not None
            and evaluation.confirmed_output > 0
            and supports_two_scoring_plays(state, card)
        ]
        if not scorers or state.score_gap <= 2 * max(
            evaluation.confirmed_output for _, evaluation in scorers
        ):
            return candidates
        best_scorer_score = max(evaluation.score for _, evaluation in scorers)
        ranked: list[tuple[CardGameObject, CardEvaluation]] = []
        for card, evaluation in candidates:
            if card.card is None or evaluation.confirmed_output > 0:
                ranked.append((card, evaluation))
                continue
            future = simulate_play(state, card.card)
            if future.playable_count > 0 or any(
                supports_two_scoring_plays(future, scorer)
                for scorer, _ in scorers
            ):
                ranked.append((card, evaluation))
                continue
            protected = replace(
                evaluation,
                score=min(evaluation.score, best_scorer_score - 1.0),
                reason=f'{evaluation.reason}；保留后续产分体力',
            )
            ranked.append((card, protected))
        return ranked

    def _protect_exam_finish(
        self,
        candidates: list[tuple['CardGameObject', CardEvaluation]],
        state: BattleState,
    ) -> list[tuple['CardGameObject', CardEvaluation]]:
        """Prefer certain output to single-play setup near an exam deadline."""
        if not state.is_exam or state.remaining_turns > 3:
            return candidates
        scorers = [
            evaluation for _, evaluation in candidates
            if evaluation.confirmed_output > 0
        ]
        if not scorers:
            return candidates
        best_scorer_score = max(evaluation.score for evaluation in scorers)
        ranked: list[tuple[CardGameObject, CardEvaluation]] = []
        for card, evaluation in candidates:
            if card.card is None or evaluation.confirmed_output > 0:
                ranked.append((card, evaluation))
                continue
            future = simulate_play(state, card.card)
            if (
                future.playable_count > 0
                or (
                    future.anomaly.full_power_pending
                    and not state.anomaly.full_power_pending
                )
                or (
                    future.anomaly.full_power_active
                    and not state.anomaly.full_power_active
                )
                or any(
                    effect.produce_exam_effect is not None
                    and effect.produce_exam_effect.effect_type
                    == ProduceExamEffectType.ExamReview
                    for effect in card.card.play_effects
                )
            ):
                ranked.append((card, evaluation))
                continue
            if state.remaining_turns == 3:
                effect_types = {
                    effect.produce_exam_effect.effect_type
                    for effect in card.card.play_effects
                    if effect.produce_exam_effect is not None
                }
                surplus_defence = (
                    state.genki_known
                    and state.genki >= 10
                    and bool(effect_types)
                    and effect_types <= {
                        ProduceExamEffectType.ExamBlock,
                        ProduceExamEffectType.ExamBlockFix,
                        ProduceExamEffectType.ExamPreservation,
                        ProduceExamEffectType.ExamOverPreservation,
                    }
                )
                if not surplus_defence:
                    ranked.append((card, evaluation))
                    continue
            ranked.append((card, replace(
                evaluation,
                score=min(evaluation.score, best_scorer_score - 1.0),
                reason=f'{evaluation.reason}；考试末段优先确定产分',
            )))
        return ranked

    def _prefer_lesson_finish_progress(
        self,
        candidates: list[tuple['CardGameObject', CardEvaluation]],
        state: BattleState,
    ) -> list[tuple['CardGameObject', CardEvaluation]]:
        """Value certain CLEAR progress near the lesson deadline.

        This is a modest bonus, not a hard scorer-only filter: a strong setup
        card may still be worth playing if it enables a better next turn. With
        three turns left, only nearly sufficient output receives a larger
        bonus, so one safe play can put CLEAR within reach.
        """
        if (
            state.is_exam
            or state.score_target_is_final is not False
            or state.score_gap is None
            or state.score_gap <= 0
            or state.remaining_turns > 3
        ):
            return candidates
        ranked: list[tuple[CardGameObject, CardEvaluation]] = []
        for card, evaluation in candidates:
            certain_progress = min(
                state.score_gap,
                conservative_target_output(evaluation.confirmed_output),
            )
            if certain_progress <= 0:
                ranked.append((card, evaluation))
                continue
            if state.remaining_turns == 3:
                if certain_progress < 0.75 * state.score_gap:
                    ranked.append((card, evaluation))
                    continue
                bonus = max(15.0, 0.8 * certain_progress)
            else:
                bonus = 0.3 * certain_progress
            ranked.append((card, replace(
                evaluation,
                score=evaluation.score + bonus,
                reason=f'{evaluation.reason}；课程末段确定产分',
            )))
        return ranked

    def _prefer_exam_catch_up_output(
        self,
        candidates: list[tuple['CardGameObject', CardEvaluation]],
        state: BattleState,
    ) -> list[tuple['CardGameObject', CardEvaluation]]:
        """While behind, favor certain output without comparing incompatible units.

        The exam standings are scaled scores, whereas confirmed_output is a
        card-effect parameter estimate. Delayed effects and duplicate cards
        prevent a reliable per-play conversion from the visible score delta.
        """
        if (
            not state.is_exam
            or state.remaining_turns > 3
            or state.current_score is None
            or state.target_score is None
        ):
            return candidates
        if state.target_score <= state.current_score:
            return candidates
        ranked: list[tuple[CardGameObject, CardEvaluation]] = []
        for card, evaluation in candidates:
            certain_output = conservative_target_output(evaluation.confirmed_output)
            if certain_output <= 0:
                ranked.append((card, evaluation))
                continue
            ranked.append((card, replace(
                evaluation,
                score=evaluation.score + min(35.0, 10.0 + 0.35 * certain_output),
                reason=f'{evaluation.reason}；考试落后时优先确定产分',
            )))
        return ranked

    def _add_shallow_lookahead(
        self,
        candidates: list[tuple['CardGameObject', CardEvaluation]],
        state: BattleState,
    ) -> list[tuple['CardGameObject', CardEvaluation]]:
        """Compare two-card orderings only when another play is available."""
        ranked: list[tuple[CardGameObject, CardEvaluation]] = []
        for first_card, first in candidates:
            if first_card.card is None:
                continue
            future = simulate_play(state, first_card.card)
            safe_output = conservative_target_output(first.confirmed_output)
            if state.score_gap is not None and safe_output > 0:
                if (
                    state.score_target_is_final is False
                    and safe_output >= state.score_gap
                ):
                    # CLEAR advances to PERFECT; its next gap is not visible
                    # until the HUD refreshes after the first card.
                    future = replace(
                        future, score_gap=None, score_target_is_final=None
                    )
                else:
                    future = replace(
                        future,
                        score_gap=max(
                            0, state.score_gap - safe_output
                        ),
                    )
            followups = [
                evaluation
                for second_card, _ in candidates
                if second_card is not first_card and second_card.card is not None
                and can_afford_play(future, second_card.card) is True
                for evaluation in [self._evaluator.evaluate(second_card.card, future)]
                if evaluation is not None
            ]
            if future.playable_count <= 0 or not followups:
                ranked.append((first_card, first))
                continue
            best_followup = max(followups, key=lambda item: item.score)
            followup_score = max(0.0, best_followup.score) * 0.9
            ranked.append((
                first_card,
                CardEvaluation(
                    card_id=first.card_id,
                    score=first.score + followup_score,
                    confidence=first.confidence,
                    reason=(
                        f'{first.reason}；两步预估后续 {best_followup.card_id}'
                        f' {followup_score:+.1f}'
                    ),
                    projected_output=first.projected_output,
                    confirmed_output=first.confirmed_output,
                ),
            ))
        return ranked

    def _build_observed_state(
        self,
        ctx: 'LessonBattleContext',
        hands: list['CardGameObject | None'],
    ) -> BattleState:
        fetch_turns = (
            getattr(ctx, 'fetch_exam_remaining_turns', ctx.fetch_remaining_turns)
            if self._is_exam else ctx.fetch_remaining_turns
        )
        observed_turns = fetch_turns()
        remaining_turns = self._memory.reconcile_remaining_turns(observed_turns)
        turns_known = (
            observed_turns is not None
            and 1 <= observed_turns <= 30
            and observed_turns == remaining_turns
        )
        cards = [
            card.card
            for card in hands
            if card is not None and card.card is not None
        ]
        if not self._memory.turn_baseline_known:
            # Neither the initial fallback nor an unaccepted first OCR jump
            # establishes an elapsed turn. Keep effects until a real baseline.
            self._memory.previous_remaining_turns = None
            if turns_known:
                self._memory.turn_baseline_known = True
        self._memory.observe(remaining_turns, cards)
        session = produce_session()
        archetype = session.archetype if session is not None else None
        profile = profile_for(archetype)
        fetch_score_gap = None if self._is_exam else getattr(ctx, 'fetch_score_gap', None)
        score_gap = fetch_score_gap() if fetch_score_gap is not None else None
        fetch_target_stage = None if self._is_exam else getattr(ctx, 'fetch_score_target_is_final', None)
        score_target_is_final = fetch_target_stage() if fetch_target_stage is not None else None
        fetch_standings = None if not self._is_exam else getattr(ctx, 'fetch_exam_standings', None)
        standings = fetch_standings() if fetch_standings is not None else None
        current_score = standings.own_score if standings is not None else None
        target_score = (
            standings.leader_score + 1
            if standings is not None and not standings.own_is_leader
            else None
        )
        if standings is not None:
            logger.debug(
                'Planner exam standings: own=%d leader=%d own_is_leader=%s',
                standings.own_score, standings.leader_score, standings.own_is_leader,
            )
        if not self._is_exam:
            target_stage = (
                'PERFECT' if score_target_is_final is True
                else 'CLEAR' if score_target_is_final is False
                else 'unknown'
            )
            logger.debug('Planner lesson target: %s, remaining score: %s', target_stage, score_gap)
        fetch_screen = getattr(ctx, 'fetch_screen', None)
        screen = fetch_screen() if fetch_screen is not None else None
        observations = observe_effect_icons(screen) if screen is not None else []
        observed_effects = frozenset(item.effect for item in observations)
        observed_values = {
            item.effect: item.value
            for item in observations
        }
        if screen is not None:
            self._memory.reconcile_effect_values(observed_values)
        if observations:
            logger.debug(
                'Planner observed HUD effects: %s',
                [
                    f'{item.effect}={item.value}'
                    if item.value is not None
                    else item.effect
                    for item in observations
                ],
            )
        fetch_hp = (
            getattr(ctx, 'fetch_exam_hp', ctx.fetch_hp)
            if self._is_exam else ctx.fetch_hp
        )
        fetch_genki = (
            getattr(ctx, 'fetch_exam_stamina', ctx.fetch_stamina)
            if self._is_exam else ctx.fetch_stamina
        )
        observed_hp = fetch_hp()
        hp = self._reconcile_hp_ocr(observed_hp)
        observed_genki = fetch_genki()
        genki = observed_genki
        return BattleState(
            remaining_turns=remaining_turns if turns_known else 1,
            remaining_turns_known=turns_known,
            observed_remaining_turns=observed_turns,
            observed_hp=observed_hp,
            observed_genki=observed_genki,
            observed_effect_values=tuple(sorted(observed_values.items())),
            hp=hp if hp is not None else 0,
            genki=genki if genki is not None else 0,
            hp_known=hp is not None,
            genki_known=genki is not None,
            current_score=current_score,
            target_score=target_score,
            score_gap=score_gap,
            score_target_is_final=score_target_is_final,
            is_exam=self._is_exam,
            playable_count=max(
                1,
                1
                + self._memory.anomaly.extra_play_count
                - self._memory.plays_this_turn,
            ),
            archetype=archetype,
            plan_type=profile.plan_type if profile is not None else None,
            hand_card_ids=tuple(
                card.card._id
                for card in hands
                if card is not None and card.card is not None
            ),
            hand_size=len(hands),
            draw_pile_card_ids=self._memory.inferred_draw_pile(cards),
            discard_card_ids=tuple(self._memory.discard),
            removed_card_ids=tuple(self._memory.removed),
            observed_effects=observed_effects,
            confirmed_absent_effects=frozenset(
                effect
                for effect, misses in self._memory.effect_misses.items()
                if misses >= 2
            ),
            anomaly=self._memory.anomaly,
            buffs=self._memory.buffs,
        )

    def _reconcile_hp_ocr(self, observed: int | None) -> int | None:
        """Correct a corroborated prefix; mark ambiguous three-digit HP unknown."""
        previous = self._last_move_state
        if (
            observed is None
            or previous is None
            or not previous.hp_known
            or not 0 <= previous.hp < 100
            or not 100 <= observed < 1000
        ):
            return observed
        suffix = observed % 100
        if abs(suffix - previous.hp) > 12:
            logger.warning(
                'Planner discarded suspect hp OCR %d (predicted %d)',
                observed, previous.hp,
            )
            return None
        logger.warning(
            'Planner corrected suspect hp OCR %d to %d (predicted %d)',
            observed, suffix, previous.hp,
        )
        return suffix
