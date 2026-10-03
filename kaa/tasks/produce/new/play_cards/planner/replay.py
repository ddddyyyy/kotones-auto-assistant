"""Screen trace samples for manual review; never treat deltas as training truth."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path


def screen_trace(path: Path) -> list[dict]:
    events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    decisions = {e['sequence']: e for e in events if e.get('event') == 'decision'}
    feedbacks = {e['sequence']: e for e in events if e.get('event') == 'feedback'}
    rows = []
    for sequence, decision in decisions.items():
        before = decision['state']
        feedback_event = feedbacks.get(sequence, {})
        feedback = feedback_event.get('feedback', {})
        after = feedback_event.get('after_state')
        source = 'feedback_frame'
        terminal_unobserved = feedback.get('evidence') == 'battle_ended_without_observation'
        if terminal_unobserved:
            after = None
            source = 'battle_end_unobserved'
        elif after is None:
            after = decisions.get(sequence + 1, {}).get('state')
            source = 'next_decision_legacy'
        reasons = ['battle_end_unobserved'] if terminal_unobserved else []
        if not feedback:
            reasons.append('missing_feedback')
        elif feedback.get('status') != 'confirmed':
            reasons.append('unconfirmed_play')
        if feedback.get('evidence') == 'card_move_dialog':
            reasons.append('move_dialog')
        if after is None:
            reasons.append('missing_after_state')
        else:
            if before.get('remaining_turns_known') is False or after.get('remaining_turns_known') is False:
                reasons.append('unknown_turns')
            if before.get('remaining_turns') != after.get('remaining_turns'):
                reasons.append('turn_changed')
            if before.get('score_target_is_final') is None or after.get('score_target_is_final') is None:
                reasons.append('unknown_target')
            elif before['score_target_is_final'] != after['score_target_is_final']:
                reasons.append('target_changed')
        delta = feedback.get('observed_gap_reduction')
        if delta is None:
            reasons.append('missing_delta')
        projected = decision['selected'].get('projected_output', 0)
        if projected <= 0:
            reasons.append('no_positive_output_prediction')
        rows.append({
            'source_file': str(path.resolve()), 'sequence': sequence,
            'after_source': source, 'schema_version': decision.get('schema_version'),
            'card_id': decision['selected']['card_id'],
            'archetype': before.get('archetype'), 'is_exam': before.get('is_exam'),
            'remaining_turns': before.get('remaining_turns'),
            'hp_known': before.get('hp_known'), 'genki_known': before.get('genki_known'),
            'projected_output': projected,
            'observed_gap_reduction': delta,
            'diagnostic_error': delta - projected if delta is not None else None,
            'bucket': 'manual_review_candidate' if not reasons else 'excluded',
            'exclusion_reasons': reasons,
            'label_status': 'unverified_requires_ocr_effect_order_enchant_review',
            'before_state': before, 'after_state': after,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--code-version', default=None, help='Provenance supplied by operator, not inferred')
    parser.add_argument('--run-id', required=True, help='Keep complete produce runs together when splitting datasets')
    parser.add_argument('--scheme', required=True)
    args = parser.parse_args()
    paths = sorted(args.directory.rglob('decisions-*.jsonl'))
    if not paths:
        parser.error('No decision trace files found')
    rows = [row for path in paths for row in screen_trace(path)]
    for row in rows:
        row.update(code_version=args.code_version, run_id=args.run_id, scheme=args.scheme)
    summary = dict(Counter(row['bucket'] for row in rows))
    report = {'summary': summary, 'caveat': 'Screening only; no clean labels, no unchosen-card outcomes, no weight fitting.', 'samples': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
