"""偶像之路只读数据入口。"""

from dataclasses import dataclass
from functools import lru_cache

from ._util import register_cache_clear
from .sqlite import select_many


@dataclass(frozen=True)
class TowerCharacter:
    id: str
    name: str


@lru_cache(maxsize=1)
def tower_characters() -> tuple[TowerCharacter, ...]:
    rows = select_many('''
        SELECT DISTINCT C.id, C.lastName || C.firstName AS name, T.[order]
        FROM Tower T JOIN Character C ON T.characterId = C.id
        ORDER BY T.[order]
    ''')
    return tuple(TowerCharacter(row['id'], row['name']) for row in rows)


register_cache_clear(tower_characters.cache_clear)
