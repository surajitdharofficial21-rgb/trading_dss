"""Tests for IndexRegistry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.index_registry import Index, IndexRegistry, IndexRegistryError, reset_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_json(entries: list[dict], tmp_path: Path) -> Path:
    p = tmp_path / "indices.json"
    p.write_text(json.dumps(entries), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Basic load / get
# ---------------------------------------------------------------------------


class TestLoadAndGet:
    def test_load_from_file(self, registry: IndexRegistry) -> None:
        assert len(registry) == 4

    def test_get_known_index(self, registry: IndexRegistry) -> None:
        idx = registry.get("NIFTY50")
        assert isinstance(idx, Index)
        assert idx.display_name == "NIFTY 50"
        assert idx.exchange == "NSE"
        assert idx.has_options is True
        assert idx.option_symbol == "NIFTY"

    def test_get_case_insensitive(self, registry: IndexRegistry) -> None:
        assert registry.get("nifty50").id == "NIFTY50"

    def test_get_unknown_raises(self, registry: IndexRegistry) -> None:
        with pytest.raises(KeyError):
            registry.get("DOESNOTEXIST")

    def test_get_index_returns_none_for_unknown(self, registry: IndexRegistry) -> None:
        assert registry.get_index("GHOST") is None

    def test_get_or_none_returns_none(self, registry: IndexRegistry) -> None:
        assert registry.get_or_none("GHOST") is None

    def test_get_index_or_raise_raises_key_error(self, registry: IndexRegistry) -> None:
        with pytest.raises(KeyError, match="Unknown index id"):
            registry.get_index_or_raise("GHOST")

    def test_contains_true(self, registry: IndexRegistry) -> None:
        assert "NIFTY50" in registry

    def test_contains_false(self, registry: IndexRegistry) -> None:
        assert "FAKEIDX" not in registry

    def test_contains_case_insensitive(self, registry: IndexRegistry) -> None:
        assert "nifty50" in registry

    def test_repr_contains_totals(self, registry: IndexRegistry) -> None:
        r = repr(registry)
        assert "total=4" in r


# ---------------------------------------------------------------------------
# Active / all indices
# ---------------------------------------------------------------------------


class TestActiveIndices:
    def test_get_all_includes_inactive(self, tmp_path: Path) -> None:
        entries = [
            {
                "id": "ACTIVE", "display_name": "Active", "exchange": "NSE",
                "has_options": False, "is_active": True,
            },
            {
                "id": "INACTIVE", "display_name": "Inactive", "exchange": "NSE",
                "has_options": False, "is_active": False,
            },
        ]
        reg = IndexRegistry.from_file(_make_json(entries, tmp_path))
        assert len(reg.get_all_indices(active_only=False)) == 2
        assert len(reg.get_all_indices(active_only=True)) == 1

    def test_get_active_indices(self, registry: IndexRegistry) -> None:
        active = registry.get_active_indices()
        assert all(i.is_active for i in active)
        assert len(active) == 4  # all 4 fixture entries are active

    def test_ids_active_only(self, registry: IndexRegistry) -> None:
        ids = registry.ids(active_only=True)
        assert "NIFTY50" in ids
        assert "BANKNIFTY" in ids


# ---------------------------------------------------------------------------
# Filter by exchange
# ---------------------------------------------------------------------------


class TestFilterByExchange:
    def test_get_by_nse(self, registry: IndexRegistry) -> None:
        nse = registry.get_indices_by_exchange("NSE")
        assert len(nse) == 3
        assert all(i.exchange == "NSE" for i in nse)

    def test_get_by_bse(self, registry: IndexRegistry) -> None:
        bse = registry.get_indices_by_exchange("BSE")
        assert len(bse) == 1
        assert bse[0].id == "SENSEX"

    def test_get_by_exchange_case_insensitive(self, registry: IndexRegistry) -> None:
        assert registry.get_indices_by_exchange("nse") == registry.get_indices_by_exchange("NSE")

    def test_filter_exchange(self, registry: IndexRegistry) -> None:
        bse = registry.filter(exchange="BSE")
        assert all(i.exchange == "BSE" for i in bse)
        assert len(bse) == 1


# ---------------------------------------------------------------------------
# Filter by options
# ---------------------------------------------------------------------------


class TestFilterByOptions:
    def test_get_indices_with_options(self, registry: IndexRegistry) -> None:
        fo = registry.get_indices_with_options()
        assert all(i.has_options for i in fo)
        assert len(fo) == 3  # NIFTY50, BANKNIFTY, SENSEX

    def test_filter_has_options_true(self, registry: IndexRegistry) -> None:
        fo = registry.filter(has_options=True)
        assert all(i.has_options for i in fo)

    def test_filter_has_options_false(self, registry: IndexRegistry) -> None:
        no_fo = registry.filter(has_options=False)
        assert all(not i.has_options for i in no_fo)
        assert len(no_fo) == 1  # NIFTY_IT

    def test_is_fo_enabled_property(self, registry: IndexRegistry) -> None:
        nifty = registry.get("NIFTY50")
        assert nifty.is_fo_enabled is True  # has_options=True, lot_size=75
        it = registry.get("NIFTY_IT")
        assert it.is_fo_enabled is False  # has_options=False


# ---------------------------------------------------------------------------
# Filter by sector
# ---------------------------------------------------------------------------


class TestFilterBySector:
    def test_get_by_broad_market(self, registry: IndexRegistry) -> None:
        broad = registry.get_indices_by_sector("broad_market")
        assert len(broad) == 2  # NIFTY50 + SENSEX
        assert all(i.sector_category == "broad_market" for i in broad)

    def test_get_by_sectoral(self, registry: IndexRegistry) -> None:
        sectoral = registry.get_indices_by_sector("sectoral")
        assert len(sectoral) == 2  # BANKNIFTY + NIFTY_IT
        assert all(i.sector_category == "sectoral" for i in sectoral)

    def test_filter_sector_category(self, registry: IndexRegistry) -> None:
        result = registry.filter(sector_category="broad_market")
        assert all(i.sector_category == "broad_market" for i in result)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearchIndices:
    def test_search_by_id_exact(self, registry: IndexRegistry) -> None:
        results = registry.search_indices("nifty50")
        assert results[0].id == "NIFTY50"

    def test_search_by_display_name_partial(self, registry: IndexRegistry) -> None:
        results = registry.search_indices("bank")
        ids = [r.id for r in results]
        assert "BANKNIFTY" in ids

    def test_search_by_option_symbol(self, registry: IndexRegistry) -> None:
        results = registry.search_indices("NIFTY")
        # Should match NIFTY50, BANKNIFTY at minimum
        ids = [r.id for r in results]
        assert "NIFTY50" in ids

    def test_search_by_sector(self, registry: IndexRegistry) -> None:
        results = registry.search_indices("broad_market")
        assert all(r.sector_category == "broad_market" for r in results)

    def test_search_empty_returns_all_active(self, registry: IndexRegistry) -> None:
        results = registry.search_indices("")
        assert len(results) == len(registry.get_active_indices())

    def test_search_no_match_returns_empty(self, registry: IndexRegistry) -> None:
        results = registry.search_indices("zzznomatch999")
        assert results == []

    def test_search_exact_id_comes_first(self, registry: IndexRegistry) -> None:
        results = registry.search_indices("SENSEX")
        assert results[0].id == "SENSEX"


# ---------------------------------------------------------------------------
# yahoo_symbols helper
# ---------------------------------------------------------------------------


class TestYahooSymbols:
    def test_returns_only_indices_with_yahoo(self, registry: IndexRegistry) -> None:
        symbols = registry.yahoo_symbols()
        assert "NIFTY50" in symbols
        assert symbols["NIFTY50"] == "^NSEI"
        # NIFTY_IT has yahoo_symbol so should be included
        assert "NIFTY_IT" in symbols

    def test_all_values_are_strings(self, registry: IndexRegistry) -> None:
        assert all(isinstance(v, str) for v in registry.yahoo_symbols().values())


# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------


class TestReload:
    def test_reload_updates_count(self, tmp_path: Path) -> None:
        initial = [
            {
                "id": "IDX1", "display_name": "Index 1", "exchange": "NSE",
                "has_options": False, "is_active": True,
            },
        ]
        p = _make_json(initial, tmp_path)
        reg = IndexRegistry.from_file(p)
        assert len(reg) == 1

        # Add a second entry and reload
        updated = initial + [
            {
                "id": "IDX2", "display_name": "Index 2", "exchange": "BSE",
                "has_options": False, "is_active": True,
            },
        ]
        p.write_text(json.dumps(updated), encoding="utf-8")
        count = reg.reload()
        assert count == 2
        assert len(reg) == 2
        assert "IDX2" in reg

    def test_reload_without_source_raises(self) -> None:
        reg = IndexRegistry(indices=[])
        with pytest.raises(IndexRegistryError, match="not loaded from a file"):
            reg.reload()


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IndexRegistryError, match="not found"):
            IndexRegistry.from_file(tmp_path / "nonexistent.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        with pytest.raises(IndexRegistryError):
            IndexRegistry.from_file(bad)

    def test_not_a_list_raises(self, tmp_path: Path) -> None:
        p = _make_json.__wrapped__ if hasattr(_make_json, "__wrapped__") else None
        bad = tmp_path / "bad.json"
        bad.write_text('{"id": "X"}')
        with pytest.raises(IndexRegistryError, match="array"):
            IndexRegistry.from_file(bad)

    def test_invalid_exchange_raises(self, tmp_path: Path) -> None:
        bad_entry = [{"id": "BAD", "display_name": "Bad", "exchange": "MCX", "has_options": False}]
        with pytest.raises(IndexRegistryError):
            IndexRegistry.from_file(_make_json(bad_entry, tmp_path))

    def test_has_options_without_symbol_raises(self, tmp_path: Path) -> None:
        bad_entry = [{
            "id": "BAD", "display_name": "Bad", "exchange": "NSE",
            "lot_size": 50, "has_options": True, "option_symbol": None,
        }]
        with pytest.raises(IndexRegistryError):
            IndexRegistry.from_file(_make_json(bad_entry, tmp_path))

    def test_negative_lot_size_raises(self, tmp_path: Path) -> None:
        bad_entry = [{
            "id": "BAD", "display_name": "Bad", "exchange": "NSE",
            "lot_size": -10, "has_options": False,
        }]
        with pytest.raises(IndexRegistryError):
            IndexRegistry.from_file(_make_json(bad_entry, tmp_path))

    def test_multiple_errors_reported(self, tmp_path: Path) -> None:
        bad_entries = [
            {"id": "BAD1", "display_name": "Bad 1", "exchange": "MCX", "has_options": False},
            {"id": "BAD2", "display_name": "Bad 2", "exchange": "XYZ", "has_options": False},
        ]
        with pytest.raises(IndexRegistryError) as exc_info:
            IndexRegistry.from_file(_make_json(bad_entries, tmp_path))
        assert "2 index entries" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Index model properties
# ---------------------------------------------------------------------------


class TestIndexModel:
    def test_is_nse(self, registry: IndexRegistry) -> None:
        assert registry.get("NIFTY50").is_nse is True
        assert registry.get("SENSEX").is_nse is False

    def test_is_bse(self, registry: IndexRegistry) -> None:
        assert registry.get("SENSEX").is_bse is True
        assert registry.get("NIFTY50").is_bse is False

    def test_frozen_model_immutable(self, registry: IndexRegistry) -> None:
        idx = registry.get("NIFTY50")
        with pytest.raises(Exception):
            idx.display_name = "hacked"  # type: ignore[misc]

    def test_id_normalised_to_uppercase(self, tmp_path: Path) -> None:
        entry = [{"id": "lowercase", "display_name": "LC", "exchange": "NSE", "has_options": False}]
        reg = IndexRegistry.from_file(_make_json(entry, tmp_path))
        assert "LOWERCASE" in reg


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_reset_clears_singleton(self, indices_json_path: Path) -> None:
        from src.data.index_registry import get_registry

        reset_registry()
        reg = get_registry(path=indices_json_path)
        assert len(reg) == 4

        reset_registry()
        reg2 = get_registry(path=indices_json_path)
        assert reg2 is not reg  # fresh instance after reset
