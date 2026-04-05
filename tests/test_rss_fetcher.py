"""Tests for src.analysis.news.rss_fetcher."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.analysis.news.rss_fetcher import RSSFetcher, RawArticle

_IST = ZoneInfo("Asia/Kolkata")


# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_FEEDS = [
    {
        "id": "test_feed",
        "source_name": "TestSource",
        "url": "https://example.com/rss.xml",
        "credibility_score": 0.85,
        "category": "market",
        "language": "en",
        "refresh_interval_seconds": 120,
        "is_active": True,
        "description": "Test feed",
    },
    {
        "id": "inactive_feed",
        "source_name": "InactiveSource",
        "url": "https://example.com/inactive.xml",
        "credibility_score": 0.5,
        "category": "general",
        "language": "en",
        "refresh_interval_seconds": 300,
        "is_active": False,
        "description": "Inactive feed",
    },
]


@pytest.fixture()
def feeds_json(tmp_path: Path) -> Path:
    p = tmp_path / "rss_feeds.json"
    p.write_text(json.dumps(SAMPLE_FEEDS), encoding="utf-8")
    return p


def _make_entry(
    title: str = "Test Article",
    summary: str = "Summary text",
    link: str = "https://example.com/article1",
    published_parsed: tuple | None = (2025, 3, 15, 10, 30, 0, 5, 74, 0),
    author: str | None = "Author",
    tags: list | None = None,
    content: list | None = None,
) -> dict:
    entry = {
        "title": title,
        "summary": summary,
        "link": link,
        "published_parsed": published_parsed,
        "author": author,
        "tags": tags or [],
        "content": content,
    }
    return entry


def _mock_parsed_feed(entries: list[dict] | None = None, bozo: bool = False):
    """Return an object that looks like feedparser.parse() output."""
    feed = MagicMock()
    feed.entries = entries if entries is not None else [_make_entry()]
    feed.bozo = bozo
    feed.bozo_exception = Exception("bad xml") if bozo else None
    return feed


# ── Tests ────────────────────────────────────────────────────────────────────


class TestRSSFetcher:
    """Unit tests for the RSSFetcher class."""

    @patch("src.analysis.news.rss_fetcher.feedparser.parse")
    def test_fetch_feed_parses_entries(self, mock_parse, feeds_json):
        """feedparser output is converted into RawArticle objects."""
        mock_parse.return_value = _mock_parsed_feed([
            _make_entry(title="Article 1", link="https://ex.com/1"),
            _make_entry(title="Article 2", link="https://ex.com/2"),
        ])

        fetcher = RSSFetcher(feeds_path=feeds_json)
        articles = fetcher.fetch_feed(SAMPLE_FEEDS[0])

        assert len(articles) == 2
        assert all(isinstance(a, RawArticle) for a in articles)
        assert articles[0].title == "Article 1"
        assert articles[0].source == "TestSource"
        assert articles[0].source_credibility == 0.85

    @patch("src.analysis.news.rss_fetcher.feedparser.parse")
    def test_pubdate_converted_to_ist(self, mock_parse, feeds_json):
        """published_parsed tuple is converted to IST datetime."""
        mock_parse.return_value = _mock_parsed_feed([
            _make_entry(published_parsed=(2025, 6, 15, 4, 0, 0, 6, 166, 0)),
        ])

        fetcher = RSSFetcher(feeds_path=feeds_json)
        articles = fetcher.fetch_feed(SAMPLE_FEEDS[0])

        pub = articles[0].published_at
        assert pub.tzinfo is not None
        # UTC 04:00 → IST 09:30
        assert pub.astimezone(_IST).hour == 9
        assert pub.astimezone(_IST).minute == 30

    @patch("src.analysis.news.rss_fetcher.feedparser.parse")
    def test_malformed_feed_with_entries_still_works(self, mock_parse, feeds_json):
        """A bozo feed that still has entries should not raise."""
        feed = _mock_parsed_feed([_make_entry()], bozo=True)
        # bozo but has entries → should work
        feed.entries = [_make_entry()]
        mock_parse.return_value = feed

        fetcher = RSSFetcher(feeds_path=feeds_json)
        articles = fetcher.fetch_feed(SAMPLE_FEEDS[0])
        assert len(articles) == 1

    @patch("src.analysis.news.rss_fetcher.feedparser.parse")
    def test_malformed_feed_no_entries_raises(self, mock_parse, feeds_json):
        """A bozo feed with zero entries should raise (triggering retry)."""
        feed = _mock_parsed_feed([], bozo=True)
        mock_parse.return_value = feed

        fetcher = RSSFetcher(feeds_path=feeds_json)
        with pytest.raises(RuntimeError, match="Feed error"):
            fetcher.fetch_feed(SAMPLE_FEEDS[0])

    @patch("src.analysis.news.rss_fetcher.feedparser.parse")
    def test_refresh_interval_skips_recent_feed(self, mock_parse, feeds_json):
        """A feed fetched recently should be skipped by fetch_all_feeds."""
        mock_parse.return_value = _mock_parsed_feed([_make_entry()])

        fetcher = RSSFetcher(feeds_path=feeds_json)
        # First fetch should work
        result1 = fetcher.fetch_all_feeds()
        assert len(result1) == 1

        # Second immediate fetch should skip (refresh_interval not reached)
        result2 = fetcher.fetch_all_feeds()
        assert len(result2) == 0

    @patch("src.analysis.news.rss_fetcher.feedparser.parse")
    def test_fetch_all_skips_inactive(self, mock_parse, feeds_json):
        """Inactive feeds are excluded from fetch_all_feeds."""
        mock_parse.return_value = _mock_parsed_feed([_make_entry()])

        fetcher = RSSFetcher(feeds_path=feeds_json)
        articles = fetcher.fetch_all_feeds()

        # Only 1 active feed → called once
        assert mock_parse.call_count == 1

    @patch("src.analysis.news.rss_fetcher.feedparser.parse")
    def test_fetch_stats_tracking(self, mock_parse, feeds_json):
        """Statistics are updated after fetching."""
        mock_parse.return_value = _mock_parsed_feed([
            _make_entry(), _make_entry(link="https://ex.com/2"),
        ])

        fetcher = RSSFetcher(feeds_path=feeds_json)
        fetcher.fetch_all_feeds()

        stats = fetcher.get_fetch_stats()
        assert stats["total_fetches"] == 1
        assert stats["successful"] == 1
        assert stats["failed"] == 0
        assert stats["articles_fetched"] == 2
        assert stats["last_fetch_time"] is not None

    @patch("src.analysis.news.rss_fetcher.feedparser.parse")
    def test_fetch_single_source(self, mock_parse, feeds_json):
        """fetch_single_source filters by source_name."""
        mock_parse.return_value = _mock_parsed_feed([_make_entry()])

        fetcher = RSSFetcher(feeds_path=feeds_json)
        articles = fetcher.fetch_single_source("TestSource")
        assert len(articles) == 1

        # Non-existent source returns empty
        articles = fetcher.fetch_single_source("NonExistent")
        assert len(articles) == 0

    @patch("src.analysis.news.rss_fetcher.feedparser.parse")
    def test_feed_failure_continues_with_next(self, mock_parse, feeds_json):
        """A single feed failure should not crash fetch_all_feeds."""
        # Use a feeds file with two active feeds
        two_active = [
            {**SAMPLE_FEEDS[0], "id": "feed_a"},
            {**SAMPLE_FEEDS[0], "id": "feed_b", "source_name": "SourceB"},
        ]
        p = feeds_json.parent / "two_feeds.json"
        p.write_text(json.dumps(two_active), encoding="utf-8")

        # Track which URL is being fetched to decide failure vs success
        def side_effect(url, **kwargs):
            if "feed_a" not in url:
                # _parse_feed doesn't pass the feed id, so we use call count:
                # feed_a retries 3 times, then feed_b succeeds
                pass
            raise RuntimeError("network error")

        # First 3 calls (feed_a retries) fail, then feed_b succeeds
        mock_parse.side_effect = [
            RuntimeError("network error"),
            RuntimeError("network error"),
            RuntimeError("network error"),
            _mock_parsed_feed([_make_entry()]),
        ]

        fetcher = RSSFetcher(feeds_path=p)
        articles = fetcher.fetch_all_feeds()

        stats = fetcher.get_fetch_stats()
        assert stats["failed"] >= 1
        # Second feed should still succeed
        assert len(articles) == 1

    @patch("src.analysis.news.rss_fetcher.feedparser.parse")
    def test_content_encoded_extracted(self, mock_parse, feeds_json):
        """content:encoded field is extracted into raw_content."""
        entry = _make_entry(
            content=[{"value": "<p>Full article body</p>", "type": "text/html"}],
        )
        mock_parse.return_value = _mock_parsed_feed([entry])

        fetcher = RSSFetcher(feeds_path=feeds_json)
        articles = fetcher.fetch_feed(SAMPLE_FEEDS[0])

        assert articles[0].raw_content == "<p>Full article body</p>"

    @patch("src.analysis.news.rss_fetcher.feedparser.parse")
    def test_tags_extracted(self, mock_parse, feeds_json):
        """RSS category tags are extracted."""
        entry = _make_entry(
            tags=[{"term": "Markets"}, {"term": "NSE"}],
        )
        mock_parse.return_value = _mock_parsed_feed([entry])

        fetcher = RSSFetcher(feeds_path=feeds_json)
        articles = fetcher.fetch_feed(SAMPLE_FEEDS[0])

        assert articles[0].tags == ["Markets", "NSE"]

    @patch("src.analysis.news.rss_fetcher.feedparser.parse")
    def test_missing_pubdate_defaults_to_now(self, mock_parse, feeds_json):
        """If published_parsed is None, use current time."""
        entry = _make_entry(published_parsed=None)
        mock_parse.return_value = _mock_parsed_feed([entry])

        fetcher = RSSFetcher(feeds_path=feeds_json)
        articles = fetcher.fetch_feed(SAMPLE_FEEDS[0])

        # Should not crash; published_at should be close to now
        assert articles[0].published_at.tzinfo is not None
