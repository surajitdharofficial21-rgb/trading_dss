"""Tests for src.analysis.news.article_parser."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from src.analysis.news.article_parser import ArticleParser, ParsedArticle
from src.analysis.news.rss_fetcher import RawArticle

_IST = ZoneInfo("Asia/Kolkata")


# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_COMPANIES = [
    {
        "name": "HDFC Bank",
        "aliases": ["hdfc bank", "hdfcbank", "hdfc"],
        "sector": "banking",
        "indices": ["NIFTY50", "BANKNIFTY", "SENSEX"],
    },
    {
        "name": "Reliance Industries",
        "aliases": ["reliance", "ril"],
        "sector": "energy",
        "indices": ["NIFTY50", "SENSEX"],
    },
    {
        "name": "TCS",
        "aliases": ["tata consultancy", "tcs"],
        "sector": "it",
        "indices": ["NIFTY50", "NIFTY_IT", "SENSEX"],
    },
]


@pytest.fixture()
def companies_json(tmp_path: Path) -> Path:
    p = tmp_path / "companies.json"
    p.write_text(json.dumps(SAMPLE_COMPANIES), encoding="utf-8")
    return p


@pytest.fixture()
def parser(companies_json: Path) -> ArticleParser:
    """Parser without an IndexRegistry (uses fallback index map)."""
    return ArticleParser(companies_path=companies_json, registry=None)


def _raw(
    title: str = "Test title",
    summary: str | None = "Summary",
    url: str = "https://example.com/art",
    source: str = "TestSource",
    published_at: datetime | None = None,
    raw_content: str | None = None,
) -> RawArticle:
    return RawArticle(
        title=title,
        summary=summary,
        url=url,
        source=source,
        source_credibility=0.85,
        category="market",
        published_at=published_at or datetime(2025, 6, 15, 11, 0, tzinfo=_IST),
        fetched_at=datetime.now(tz=_IST),
        raw_content=raw_content,
    )


# ── Text cleaning tests ─────────────────────────────────────────────────────


class TestCleanText:
    """Tests for ArticleParser._clean_text."""

    def test_strips_html_tags(self, parser):
        result = parser._clean_text("<p>Hello <b>world</b></p>")
        assert result == "Hello world"

    def test_decodes_html_entities(self, parser):
        result = parser._clean_text("AT&amp;T profits &gt; expected")
        assert "AT&T" in result
        assert ">" in result

    def test_removes_urls(self, parser):
        result = parser._clean_text("Visit https://example.com/foo for more")
        assert "https://" not in result
        assert "Visit" in result

    def test_normalizes_whitespace(self, parser):
        result = parser._clean_text("too   many    spaces\n\nnewlines")
        assert "  " not in result
        assert "\n" not in result

    def test_truncates_long_text(self, parser):
        long_text = "word " * 1000
        result = parser._clean_text(long_text)
        assert len(result) <= 2000

    def test_empty_input(self, parser):
        assert parser._clean_text("") == ""
        assert parser._clean_text(None) == ""


# ── Company extraction tests ────────────────────────────────────────────────


class TestExtractCompanies:
    """Tests for company name matching."""

    def test_matches_full_name(self, parser):
        result = parser._extract_companies("HDFC Bank reported strong Q3 results")
        assert "HDFC Bank" in result

    def test_matches_alias(self, parser):
        result = parser._extract_companies("RIL shares surged 5% today")
        assert "Reliance Industries" in result

    def test_case_insensitive(self, parser):
        result = parser._extract_companies("tcs announced a buyback")
        assert "TCS" in result

    def test_multiple_companies(self, parser):
        text = "HDFC Bank and Reliance Industries led the rally"
        result = parser._extract_companies(text)
        assert "HDFC Bank" in result
        assert "Reliance Industries" in result

    def test_no_match(self, parser):
        result = parser._extract_companies("Weather forecast for Mumbai")
        assert result == []


# ── Sector extraction tests ─────────────────────────────────────────────────


class TestExtractSectors:
    """Tests for sector keyword matching."""

    def test_banking_sector(self, parser):
        result = parser._extract_sectors("Banking sector rallied today")
        assert "banking" in result

    def test_multiple_sectors(self, parser):
        result = parser._extract_sectors("IT and pharma stocks were mixed")
        assert "it" in result
        assert "pharma" in result

    def test_no_sector(self, parser):
        result = parser._extract_sectors("Sunny weather in Delhi")
        assert result == []


# ── Index extraction tests ───────────────────────────────────────────────────


class TestExtractIndices:
    """Tests for index name matching (using fallback map)."""

    def test_nifty_match(self, parser):
        result = parser._extract_indices("Nifty hits all-time high")
        assert any("NIFTY" in idx.upper() for idx in result)

    def test_sensex_match(self, parser):
        result = parser._extract_indices("Sensex drops 300 points")
        assert any("SENSEX" in idx.upper() for idx in result)

    def test_bank_nifty_match(self, parser):
        result = parser._extract_indices("Bank Nifty surged past 50000")
        assert any("BANK" in idx.upper() for idx in result)

    def test_no_index(self, parser):
        result = parser._extract_indices("Rain expected in Kerala")
        assert result == []


# ── Event type classification ────────────────────────────────────────────────


class TestClassifyEventType:
    """Tests for event type classification."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("TCS quarterly results beat estimates", "EARNINGS"),
            ("RBI monetary policy decision tomorrow", "POLICY"),
            ("India GDP growth at 7.2%", "MACRO"),
            ("US Fed signals rate pause, global markets rally", "GLOBAL"),
            ("Reliance announces mega merger deal", "CORPORATE"),
            ("SEBI imposes penalty on broker", "REGULATORY"),
            ("Nifty crashes 500 points in one hour", "MARKET_MOVE"),
            ("FII sold Rs 5000 crore in cash segment", "FII_DII"),
            ("Sunny day in Mumbai", None),
        ],
    )
    def test_classification(self, text, expected):
        result = ArticleParser._classify_event_type(text)
        assert result == expected

    def test_first_matching_rule_wins(self):
        """If text matches multiple event types, the first rule wins."""
        text = "RBI rate cut boosts GDP forecast"
        result = ArticleParser._classify_event_type(text)
        # POLICY comes before MACRO in the rule list
        assert result == "POLICY"


# ── Market hours relevance ───────────────────────────────────────────────────


class TestMarketHoursRelevance:
    """Tests for _is_market_hours_relevant."""

    def test_during_market_hours(self):
        # Wednesday 11:00 IST → market hours
        dt = datetime(2025, 6, 18, 11, 0, tzinfo=_IST)
        assert ArticleParser._is_market_hours_relevant(dt, None) is True

    def test_before_market_hours(self):
        # Wednesday 08:00 IST → before market
        dt = datetime(2025, 6, 18, 8, 0, tzinfo=_IST)
        assert ArticleParser._is_market_hours_relevant(dt, None) is False

    def test_after_close_global_event(self):
        # Wednesday 20:00 IST, GLOBAL event → relevant for next session
        dt = datetime(2025, 6, 18, 20, 0, tzinfo=_IST)
        assert ArticleParser._is_market_hours_relevant(dt, "GLOBAL") is True

    def test_after_close_earnings_event(self):
        # Wednesday 20:00 IST, EARNINGS event → not relevant
        dt = datetime(2025, 6, 18, 20, 0, tzinfo=_IST)
        assert ArticleParser._is_market_hours_relevant(dt, "EARNINGS") is False

    def test_weekend_macro_event(self):
        # Saturday, MACRO event → relevant (affects Monday)
        dt = datetime(2025, 6, 14, 10, 0, tzinfo=_IST)
        assert ArticleParser._is_market_hours_relevant(dt, "MACRO") is True

    def test_weekend_no_event(self):
        # Saturday, no event type → not relevant
        dt = datetime(2025, 6, 14, 10, 0, tzinfo=_IST)
        assert ArticleParser._is_market_hours_relevant(dt, None) is False


# ── Full parse pipeline ─────────────────────────────────────────────────────


class TestParseArticle:
    """Integration tests for the full parse_article pipeline."""

    def test_full_parse(self, parser):
        raw = _raw(
            title="HDFC Bank Q3 results: profit rises 20%",
            summary="The banking giant reported strong earnings.",
        )
        result = parser.parse_article(raw)

        assert result is not None
        assert isinstance(result, ParsedArticle)
        assert result.title == "HDFC Bank Q3 results: profit rises 20%"
        assert "HDFC Bank" in result.mentioned_companies
        assert "banking" in result.mentioned_sectors
        assert result.event_type == "EARNINGS"
        assert result.word_count > 0

    def test_html_in_raw_content(self, parser):
        raw = _raw(
            title="Test",
            summary="<p>HTML <b>content</b> &amp; entities</p>",
        )
        result = parser.parse_article(raw)
        assert "&amp;" not in result.clean_text
        assert "<p>" not in result.clean_text

    def test_empty_article_returns_none(self, parser):
        raw = _raw(title="", summary=None, raw_content=None)
        result = parser.parse_article(raw)
        assert result is None
