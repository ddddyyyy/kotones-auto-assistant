from kaa.db import skill_card, _util


def test_category_cache_is_invalidated_with_game_data(monkeypatch):
    rows = {'asset': {'category': 'ProduceCardCategory_ActiveSkill'}}
    calls = []
    def query(sql, asset):
        calls.append(asset)
        return rows.get(asset)
    monkeypatch.setattr(skill_card, 'select', query)
    monkeypatch.setattr(_util, 'invalidate_connections', lambda: None)
    skill_card.category_from_asset_id.cache_clear()
    try:
        assert skill_card.category_from_asset_id('asset') == 'ProduceCardCategory_ActiveSkill'
        assert skill_card.category_from_asset_id('missing') is None
        rows['asset'] = {'category': 'ProduceCardCategory_MentalSkill'}
        rows['missing'] = {'category': 'ProduceCardCategory_Trouble'}
        assert skill_card.category_from_asset_id('asset') == 'ProduceCardCategory_ActiveSkill'
        assert skill_card.category_from_asset_id('missing') is None
        assert calls == ['asset', 'missing']
        _util.clear_db_caches()
        assert skill_card.category_from_asset_id('asset') == 'ProduceCardCategory_MentalSkill'
        assert skill_card.category_from_asset_id('missing') == 'ProduceCardCategory_Trouble'
        assert calls == ['asset', 'missing', 'asset', 'missing']
    finally:
        skill_card.category_from_asset_id.cache_clear()


def test_title_identity_requires_unique_name_and_category(monkeypatch):
    rows = [
        {'name': '輝け！', 'category': 'A', 'assetId': 'shine'},
        {'name': '輝け!', 'category': 'A', 'assetId': 'shine'},
        {'name': '同名', 'category': 'A', 'assetId': 'one'},
        {'name': '同名', 'category': 'A', 'assetId': 'two'},
        {'name': '同名', 'category': 'M', 'assetId': 'mental'},
    ]
    monkeypatch.setattr(skill_card, 'select_many', lambda sql: rows)
    skill_card._assets_by_title.cache_clear()
    try:
        assert skill_card.asset_from_card_title('輝け!++', 'A') == 'shine'
        assert skill_card.asset_from_card_title('輝け！＋', 'A') == 'shine'
        assert skill_card.asset_from_card_title('輝け!', 'M') is None
        assert skill_card.asset_from_card_title('輝け', 'A') is None
        assert skill_card.asset_from_card_title('同名', 'A') is None
        assert skill_card.asset_from_card_title('同名', 'M') == 'mental'
        assert skill_card.asset_from_card_title('未知', 'A') is None
    finally:
        skill_card._assets_by_title.cache_clear()


def test_title_identity_cache_refreshes_positive_negative_and_ambiguous(monkeypatch):
    rows = [
        {'name': '旧', 'category': 'A', 'assetId': 'old'},
        {'name': '同名', 'category': 'A', 'assetId': 'one'},
        {'name': '同名', 'category': 'A', 'assetId': 'two'},
    ]
    monkeypatch.setattr(skill_card, 'select_many', lambda sql: rows.copy())
    monkeypatch.setattr(_util, 'invalidate_connections', lambda: None)
    skill_card._assets_by_title.cache_clear()
    try:
        assert skill_card.asset_from_card_title('旧', 'A') == 'old'
        assert skill_card.asset_from_card_title('新', 'A') is None
        assert skill_card.asset_from_card_title('同名', 'A') is None
        rows[:] = [
            {'name': '新', 'category': 'A', 'assetId': 'new'},
            {'name': '同名', 'category': 'A', 'assetId': 'one'},
        ]
        assert skill_card.asset_from_card_title('旧', 'A') == 'old'
        assert skill_card.asset_from_card_title('新', 'A') is None
        assert skill_card.asset_from_card_title('同名', 'A') is None
        _util.clear_db_caches()
        assert skill_card.asset_from_card_title('旧', 'A') is None
        assert skill_card.asset_from_card_title('新', 'A') == 'new'
        assert skill_card.asset_from_card_title('同名', 'A') == 'one'
    finally:
        skill_card._assets_by_title.cache_clear()
