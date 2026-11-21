"""可见集中阈值／追加出牌规则；只解释页面声明，不执行事件。"""

from dataclasses import dataclass
import re
from typing import Literal
import unicodedata

from .rules import RuleTextRead
from .rule_symbols import RuleSymbolEvidence as RuleIconEvidence


@dataclass(frozen=True)
class ConditionalPlayEvidence:
    frame_id: str
    icons: tuple[RuleIconEvidence, ...]
    text_reads: tuple[RuleTextRead, ...]
    text_source: Literal['icons_masked_current_crop'] = 'icons_masked_current_crop'


@dataclass(frozen=True)
class ConditionalPlayDeclaration:
    start_turn: int
    concentration_at_least: int
    extra_plays: Literal[1]
    executable: Literal[False] = False


@dataclass(frozen=True)
class ConditionalPlayAnalysis:
    evidence: ConditionalPlayEvidence
    declaration: ConditionalPlayDeclaration | None
    status: Literal['confirmed_visible_declaration', 'turn_unknown', 'icons_unknown', 'text_unknown',
                    'unsupported_text', 'high_confidence_conflict']


def _compact(text: str) -> str:
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))


def interpret_conditional_play(turn: int | None, evidence: ConditionalPlayEvidence,
                               original_reads: tuple[RuleTextRead, ...]) -> ConditionalPlayAnalysis:
    def result(status, declaration=None):
        return ConditionalPlayAnalysis(evidence, declaration, status)

    if type(turn) is not int or turn < 1:
        return result('turn_unknown')
    icons = sorted(evidence.icons, key=lambda icon: icon.rect[0])
    if (not evidence.frame_id or len(icons) != 2
        or [icon.kind for icon in icons] != ['concentration', 'extra_play']
        or any(icon.frame_id != evidence.frame_id or not .97 <= icon.score <= 1
               or min(icon.rect) < 0 or icon.rect[2] == 0 or icon.rect[3] == 0 for icon in icons)
        or icons[0].rect[0]+icons[0].rect[2] > icons[1].rect[0]
        or abs(icons[0].rect[1]-icons[1].rect[1]) > 3):
        return result('icons_unknown')
    reads = evidence.text_reads
    if (len(reads) != 2 or any(read.frame_id != evidence.frame_id for read in reads)
        or {read.representation for read in reads} != {'original', 'grayscale'}):
        return result('text_unknown')
    texts = []
    for read in reads:
        if not 1 <= len(read.lines) <= 3 or any(not .98 <= line.confidence <= 1 for line in read.lines):
            return result('text_unknown')
        texts.append(_compact(''.join(line.text for line in read.lines)))
    if texts[0] != texts[1]:
        return result('text_unknown')
    # 首批只支持图标明确写着 +1 的规则。OCR 可能漏掉细小加号；
    # 必须同时有已确认的 +1 图标和完整“使用数追加1／+1”文字。
    match = re.fullmatch(r'集中が([1-9][0-9]*)以上の場合、スキルカード使用数追加\+?1', texts[0])
    if match is None:
        return result('unsupported_text')
    threshold = int(match[1])
    if (len(original_reads) != 2 or any(read.frame_id != evidence.frame_id for read in original_reads)
        or {read.representation for read in original_reads} != {'original', 'grayscale'}):
        return result('text_unknown')
    for read in original_reads:
        for line in read.lines:
            if not .98 <= line.confidence <= 1:
                continue
            raw = _compact(line.text)
            condition = re.search(r'集中が([1-9][0-9]*)(以上|以下|未満|超)', raw)
            gain = re.search(r'追加([+-]?)([1-9][0-9]*)', raw)
            if ((condition is not None and (int(condition[1]) != threshold or condition[2] != '以上'))
                or (gain is not None and (gain[1] == '-' or int(gain[2]) != 1))):
                return result('high_confidence_conflict')
    return result('confirmed_visible_declaration', ConditionalPlayDeclaration(turn, threshold, 1))
