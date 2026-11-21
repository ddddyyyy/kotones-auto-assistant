"""条件弹窗提供的目标描述，不是成绩、通关或奖励证据。"""

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class StarRequirement:
    stars: int | None
    text: str | None
    rank_at_most: int | None


def read_rank_requirement(stars: int | None, text: str | None) -> StarRequirement:
    if stars is not None and (type(stars) is not int or not 1 <= stars <= 3):
        raise ValueError('requirement stars must be 1, 2, 3 or None')
    normalized = re.sub(r'\s+', '', unicodedata.normalize('NFKC', text or ''))
    match = re.fullmatch(r'[・•]?([1-9][0-9]*)位(以上)?でクリア', normalized)
    rank = None
    if match is not None and (match[1] == '1' or match[2] is not None):
        rank = int(match[1])
    return StarRequirement(stars, text, rank)


@dataclass(frozen=True)
class StageObjectives:
    character_id: str
    stage: int
    requirements: tuple[StarRequirement, ...]
    source: str = 'conditions_dialog'
    identity_source: str = 'opened_from_confirmed_formation'

    def rank_limit(self, stars: int) -> int | None:
        """只返回该星档明确写出的名次上限，不由名次预测实际获星。"""
        if type(stars) is not int or not 1 <= stars <= 3:
            raise ValueError('stars must be 1, 2 or 3')
        if len(self.requirements) != 3 or {row.stars for row in self.requirements} != {1, 2, 3}:
            return None
        return next(row.rank_at_most for row in self.requirements if row.stars == stars)
