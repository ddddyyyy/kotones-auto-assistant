"""Exact full-description binding for same-name items; no trigger prediction."""

from dataclasses import dataclass
import json
from typing import Literal

from .items import ItemBindingStatus, ItemReference, confirmed_item_name, normalize_item_name


@dataclass(frozen=True)
class ItemDescriptionLine:
    text: str
    confidence: float | None


@dataclass(frozen=True)
class ItemDescriptionRead:
    frame_id: str
    representation: Literal['original', 'grayscale']
    lines: tuple[ItemDescriptionLine, ...]


@dataclass(frozen=True)
class ItemDescriptionCandidate:
    id: str
    name: str
    description_json: str | None


@dataclass(frozen=True)
class ItemDescriptionBinding:
    reference: ItemReference
    reads: tuple[ItemDescriptionRead, ...]
    status: Literal['unknown_description', 'candidate_conflict', 'no_exact_match', 'ambiguous', 'resolved']
    candidate_ids: tuple[str, ...] = ()
    item_id: str | None = None
    confirmed_description: str | None = None
    scope: Literal['description_identity_only'] = 'description_identity_only'


def database_description_text(raw: str | None) -> str | None:
    """Only declared text and the known no-wrap tag; no fuzzy correction."""
    try:
        entries = json.loads(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None
    if (not isinstance(entries, list) or not entries
        or any(not isinstance(entry, dict) or not isinstance(entry.get('text'), str) for entry in entries)):
        return None
    text = ''.join(entry['text'] for entry in entries).replace('<nobr>', '').replace('</nobr>', '')
    if '<' in text or '>' in text:
        return None
    return normalize_item_name(text) or None


def bind_item_description(reference: ItemReference, reads: tuple[ItemDescriptionRead, ...],
                          candidates: tuple[ItemDescriptionCandidate, ...] | None) -> ItemDescriptionBinding:
    """Require full dual reads from the name frame and all original candidates."""
    def result(status, ids=(), text=None):
        return ItemDescriptionBinding(reference, reads, status, ids,
                                      ids[0] if status == 'resolved' else None, text)

    name = confirmed_item_name(reference.name_reads)
    if (reference.status != ItemBindingStatus.AMBIGUOUS or name is None
        or reference.confirmed_name != name or reference.item_id is not None
        or len(reference.candidate_ids) < 2 or len(reads) != 2
        or {read.representation for read in reads} != {'original', 'grayscale'}):
        return result('unknown_description')
    texts = []
    for read in reads:
        if read.frame_id != reference.name_reads[0].frame_id or not read.lines:
            return result('unknown_description')
        for line in read.lines:
            confidence = line.confidence
            if (not isinstance(line.text, str) or not normalize_item_name(line.text)
                or confidence is None or type(confidence) not in (int, float) or not 0.98 <= confidence <= 1):
                return result('unknown_description')
        texts.append(normalize_item_name(''.join(line.text for line in read.lines)))
    if texts[0] != texts[1]:
        return result('unknown_description')
    if (candidates is None or tuple(candidate.id for candidate in candidates) != reference.candidate_ids
        or any(not candidate.id or normalize_item_name(candidate.name) != name for candidate in candidates)):
        return result('candidate_conflict')
    descriptions = tuple(database_description_text(candidate.description_json) for candidate in candidates)
    if any(text is None for text in descriptions):
        return result('candidate_conflict')
    ids = tuple(candidate.id for candidate, text in zip(candidates, descriptions) if text == texts[0])
    return result('no_exact_match' if not ids else 'resolved' if len(ids) == 1 else 'ambiguous', ids, texts[0])
