import sqlite3

from kaa.db import tower
from kaa.db._util import clear_db_caches


def test_tower_characters_use_database_order_and_cache_invalidation(monkeypatch):
    connection = sqlite3.connect(':memory:')
    connection.row_factory = sqlite3.Row
    connection.executescript('''
        CREATE TABLE Character (id TEXT, lastName TEXT, firstName TEXT);
        CREATE TABLE Tower (characterId TEXT, [order] INTEGER);
        INSERT INTO Character VALUES ('hski', '花海', '咲季'), ('hmsz', '秦谷', '美鈴'), ('npc0', 'NPC', '');
        INSERT INTO Tower VALUES ('hmsz', 2), ('hski', 1);
    ''')
    monkeypatch.setattr(tower, 'select_many', lambda query: connection.execute(query).fetchall())
    tower.tower_characters.cache_clear()
    try:
        assert [(item.id, item.name) for item in tower.tower_characters()] == [
            ('hski', '花海咲季'), ('hmsz', '秦谷美鈴'),
        ]
        connection.execute("UPDATE Character SET firstName='更新' WHERE id='hmsz'")
        assert tower.tower_characters()[1].name == '秦谷美鈴'
        clear_db_caches()
        assert tower.tower_characters()[1].name == '秦谷更新'
    finally:
        tower.tower_characters.cache_clear()
        connection.close()
