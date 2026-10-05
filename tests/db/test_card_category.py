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
