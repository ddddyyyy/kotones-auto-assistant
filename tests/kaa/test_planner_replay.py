import json
from pathlib import Path
from kaa.tasks.produce.new.play_cards.planner.replay import screen_trace


def _events(*, turn_after=2, final_after=True, known=True, evidence='card_left_hand'):
    before = dict(remaining_turns=2, remaining_turns_known=known, score_target_is_final=True)
    after = dict(remaining_turns=turn_after, remaining_turns_known=known, score_target_is_final=final_after)
    return [
        dict(event='decision', sequence=1, state=before, selected=dict(card_id='output', projected_output=12)),
        dict(event='feedback', sequence=1, feedback=dict(status='confirmed', evidence=evidence, observed_gap_reduction=10), after_state=after),
    ]


def _write(tmp_path, events):
    path = tmp_path / 'decisions-test.jsonl'
    path.write_text('\n'.join(json.dumps(e) for e in events), encoding='utf-8')
    return path


def test_screening_preserves_sources_but_never_certifies_labels(tmp_path):
    rows = screen_trace(_write(tmp_path, _events()))
    assert rows[0]['bucket'] == 'manual_review_candidate'
    assert rows[0]['after_source'] == 'feedback_frame'
    assert rows[0]['diagnostic_error'] == -2
    assert rows[0]['label_status'].startswith('unverified')


def test_screening_rejects_cross_turn_unknown_target_and_dialog(tmp_path):
    for kwargs, expected in [
        ({'turn_after': 1}, 'turn_changed'),
        ({'known': False}, 'unknown_turns'),
        ({'final_after': False}, 'target_changed'),
        ({'final_after': None}, 'unknown_target'),
        ({'evidence': 'card_move_dialog'}, 'move_dialog'),
    ]:
        row = screen_trace(_write(tmp_path, _events(**kwargs)))[0]
        assert row['bucket'] == 'excluded'
        assert expected in row['exclusion_reasons']


def test_legacy_next_decision_and_missing_final_feedback_are_explicit(tmp_path):
    events = _events()
    after = events[1].pop('after_state')
    events.append(dict(event='decision', sequence=2, state=after, selected=dict(card_id='end', projected_output=20)))
    rows = screen_trace(_write(tmp_path, events))
    assert rows[0]['after_source'] == 'next_decision_legacy'
    assert rows[1]['bucket'] == 'excluded'
    assert 'missing_feedback' in rows[1]['exclusion_reasons']


def test_terminal_unobserved_feedback_does_not_borrow_a_later_decision(tmp_path):
    events = _events()
    events[1]['feedback'].update(status='unconfirmed', evidence='battle_ended_without_observation', observed_gap_reduction=None)
    events[1]['after_state'] = None
    events.append(dict(event='decision', sequence=2, state=events[0]['state'], selected=dict(card_id='later', projected_output=20)))
    row = screen_trace(_write(tmp_path, events))[0]
    assert row['after_source'] == 'battle_end_unobserved'
    assert row['after_state'] is None
    assert row['bucket'] == 'excluded'
    assert 'battle_end_unobserved' in row['exclusion_reasons']
