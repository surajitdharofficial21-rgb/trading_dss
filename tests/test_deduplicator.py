"""Tests for src.analysis.news.deduplicator."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from src.analysis.news.article_parser import ParsedArticle
from src.analysis.news.deduplicator import ArticleDeduplicator, _jaccard, _title_words

_IST = ZoneInfo("Asia/Kolkata")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _article(
    title: str = "Default Title",
    url: str = "https://example.com/default",
    source_credibility: float = 0.8,
    published_at: datetime | None = None,
) -> ParsedArticle:
    return ParsedArticle(
        title=title,
        clean_text=title,
        url=url,
        source="TestSource",
        source_credibility=source_credibility,
        category="market",
        published_at=published_at or datetime(2025, 6, 15, 11, 0, tzinfo=_IST),
        fetched_at=datetime.now(tz=_IST),
        word_count=len(title.split()),
    )


# ── Utility function tests ──────────────────────────────────────────────────


class TestJaccard:
    """Tests for the Jaccard similarity function."""

    def test_identical_sets(self):
        assert _jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        # {"a", "b", "c"} ∩ {"b", "c", "d"} = {"b", "c"} → 2/4 = 0.5
        assert _jaccard({"a", "b", "c"}, {"b", "c", "d"}) == 0.5

    def test_empty_sets(self):
        assert _jaccard(set(), set()) == 0.0
        assert _jaccard({"a"}, set()) == 0.0


class TestTitleWords:
    """Tests for title word extraction."""

    def test_basic(self):
        assert _title_words("Nifty Hits 23000") == {"nifty", "hits", "23000"}

    def test_punctuation_removed(self):
        assert _title_words("Hello, World!") == {"hello", "world"}


# ── Duplicate detection tests ────────────────────────────────────────────────


class TestIsDuplicate:
    """Tests for single-article duplicate detection."""

    def test_exact_url_duplicate(self):
        dedup = ArticleDeduplicator()
        a1 = _article(url="https://example.com/same")
        a2 = _article(title="Completely different title", url="https://example.com/same")
        assert dedup.is_duplicate(a2, [a1]) is True

    def test_similar_title_duplicate(self):
        dedup = ArticleDeduplicator(threshold=0.6)
        a1 = _article(title="Nifty hits 23000 for first time")
        a2 = _article(
            title="Nifty crosses 23000 mark for first time",
            url="https://other.com/different",
        )
        # Jaccard = 5/8 = 0.625 > 0.6 → duplicate
        assert dedup.is_duplicate(a2, [a1]) is True

    def test_dissimilar_titles_not_duplicate(self):
        dedup = ArticleDeduplicator(threshold=0.7)
        a1 = _article(title="Nifty hits 23000")
        a2 = _article(
            title="RBI holds interest rates steady",
            url="https://other.com/different",
        )
        assert dedup.is_duplicate(a2, [a1]) is False

    def test_similar_title_outside_time_window(self):
        """Same-ish title but 12 hours apart → not duplicate."""
        dedup = ArticleDeduplicator(threshold=0.7, time_window_hours=6)
        now = datetime(2025, 6, 15, 11, 0, tzinfo=_IST)
        a1 = _article(title="Nifty surges past 23000", published_at=now)
        a2 = _article(
            title="Nifty surges past 23000 again",
            url="https://other.com/x",
            published_at=now + timedelta(hours=12),
        )
        assert dedup.is_duplicate(a2, [a1]) is False

    def test_custom_threshold(self):
        dedup = ArticleDeduplicator()
        a1 = _article(title="Market rallies on FII buying")
        a2 = _article(
            title="Market rallies strongly on FII buying spree",
            url="https://other.com/x",
        )
        # Should be duplicate at 0.5 threshold
        assert dedup.is_duplicate(a2, [a1], threshold=0.5) is True


# ── Batch deduplication tests ────────────────────────────────────────────────


class TestDeduplicateBatch:
    """Tests for batch deduplication."""

    def test_removes_url_duplicates(self):
        dedup = ArticleDeduplicator()
        articles = [
            _article(title="A", url="https://ex.com/1", source_credibility=0.9),
            _article(title="B", url="https://ex.com/1", source_credibility=0.7),
            _article(title="C", url="https://ex.com/2", source_credibility=0.8),
        ]
        result = dedup.deduplicate_batch(articles)
        urls = [a.url for a in result]
        assert urls.count("https://ex.com/1") == 1
        assert "https://ex.com/2" in urls

    def test_keeps_highest_credibility(self):
        dedup = ArticleDeduplicator()
        articles = [
            _article(
                title="Nifty hits 23000 mark today",
                url="https://low.com/1",
                source_credibility=0.6,
            ),
            _article(
                title="Nifty hits 23000 mark today",
                url="https://high.com/1",
                source_credibility=0.95,
            ),
        ]
        result = dedup.deduplicate_batch(articles)
        assert len(result) == 1
        assert result[0].source_credibility == 0.95

    def test_no_duplicates_keeps_all(self):
        dedup = ArticleDeduplicator()
        articles = [
            _article(title="Banking stocks rally", url="https://ex.com/1"),
            _article(title="IT sector faces headwinds", url="https://ex.com/2"),
            _article(title="RBI holds rates steady", url="https://ex.com/3"),
        ]
        result = dedup.deduplicate_batch(articles)
        assert len(result) == 3

    def test_empty_input(self):
        dedup = ArticleDeduplicator()
        assert dedup.deduplicate_batch([]) == []


# ── DB deduplication tests ───────────────────────────────────────────────────


class TestIsDuplicateInDB:
    """Tests for database-level deduplication."""

    def test_url_exists_in_db(self):
        dedup = ArticleDeduplicator()
        db = MagicMock()
        db.fetch_one.return_value = {"id": 1}

        article = _article(url="https://example.com/existing")
        assert dedup.is_duplicate_in_db(article, db) is True
        db.fetch_one.assert_called_once()

    def test_url_not_in_db_title_match(self):
        dedup = ArticleDeduplicator(threshold=0.7)
        db = MagicMock()
        db.fetch_one.return_value = None
        db.fetch_all.return_value = [
            {"title": "Nifty hits 23000 for first time", "published_at": "2025-06-15T11:00:00"},
        ]

        article = _article(title="Nifty crosses 23000 for first time")
        assert dedup.is_duplicate_in_db(article, db) is True

    def test_not_duplicate_in_db(self):
        dedup = ArticleDeduplicator()
        db = MagicMock()
        db.fetch_one.return_value = None
        db.fetch_all.return_value = [
            {"title": "Completely different news topic", "published_at": "2025-06-15T11:00:00"},
        ]

        article = _article(title="RBI announces new policy framework")
        assert dedup.is_duplicate_in_db(article, db) is False
