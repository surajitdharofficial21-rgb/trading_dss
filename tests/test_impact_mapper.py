"""
Tests for the NewsImpactMapper — news-to-index mapping and severity classification.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from zoneinfo import ZoneInfo

from src.analysis.news.article_parser import ParsedArticle
from src.analysis.news.impact_mapper import (
    IndexImpact,
    MappedArticle,
    MarketNewsSummary,
    NewsImpactMapper,
    _SEVERITY_ORDER,
)
from src.analysis.news.sentiment_analyzer import SentimentResult

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _now_ist() -> datetime:
    return datetime.now(tz=_IST)


def _make_article(
    title: str = "Test article",
    clean_text: str = "",
    mentioned_companies: list[str] | None = None,
    mentioned_sectors: list[str] | None = None,
    mentioned_indices: list[str] | None = None,
    event_type: str | None = None,
    source_credibility: float = 0.8,
    published_at: datetime | None = None,
) -> ParsedArticle:
    return ParsedArticle(
        title=title,
        clean_text=clean_text or title,
        url="https://example.com/test",
        source="test_source",
        source_credibility=source_credibility,
        category="market",
        published_at=published_at or _now_ist(),
        fetched_at=_now_ist(),
        mentioned_companies=mentioned_companies or [],
        mentioned_sectors=mentioned_sectors or [],
        mentioned_indices=mentioned_indices or [],
        event_type=event_type,
        is_market_hours_relevant=True,
        language="en",
        word_count=len((clean_text or title).split()),
    )


def _make_sentiment(
    adjusted_score: float = 0.5,
    confidence: float = 0.7,
    label: str = "BULLISH",
) -> SentimentResult:
    return SentimentResult(
        raw_vader_score=adjusted_score * 0.8,
        keyword_boost=0.1,
        source_weight=0.9,
        adjusted_score=adjusted_score,
        sentiment_label=label,
        confidence=confidence,
    )


@pytest.fixture()
def mapper(tmp_path: Path) -> NewsImpactMapper:
    """Create a mapper with real config files from the project."""
    project_root = Path(__file__).resolve().parent.parent
    news_mappings = project_root / "config" / "news_mappings.json"
    companies = project_root / "config" / "companies.json"
    return NewsImpactMapper(
        news_mappings_path=news_mappings,
        companies_path=companies,
        registry=None,
    )


@pytest.fixture()
def mapper_with_registry(tmp_path: Path, registry) -> NewsImpactMapper:
    """Create a mapper with a test IndexRegistry."""
    project_root = Path(__file__).resolve().parent.parent
    news_mappings = project_root / "config" / "news_mappings.json"
    companies = project_root / "config" / "companies.json"
    return NewsImpactMapper(
        news_mappings_path=news_mappings,
        companies_path=companies,
        registry=registry,
    )


# ---------------------------------------------------------------------------
# Strategy 1: Direct index mention
# ---------------------------------------------------------------------------

class TestDirectMention:
    def test_bank_nifty_direct_mention(self, mapper: NewsImpactMapper) -> None:
        """Article mentioning Bank Nifty maps to BANKNIFTY with high confidence."""
        article = _make_article(
            title="Bank Nifty surges 2% on strong banking results",
            clean_text="Bank Nifty surges 2% on strong banking results today",
            mentioned_indices=["NIFTY BANK"],
        )
        sentiment = _make_sentiment(adjusted_score=0.6)
        impacts = mapper.map_to_indices(article, sentiment)

        banknifty = [i for i in impacts if i.index_id == "BANKNIFTY"]
        assert len(banknifty) >= 1
        assert banknifty[0].confidence >= 0.9
        assert "NIFTY BANK" in banknifty[0].affected_by

    def test_multiple_direct_mentions(self, mapper: NewsImpactMapper) -> None:
        """Article mentioning multiple indices maps all of them."""
        article = _make_article(
            title="Nifty and Sensex rally",
            mentioned_indices=["NIFTY 50", "BSE SENSEX"],
        )
        sentiment = _make_sentiment(adjusted_score=0.5)
        impacts = mapper.map_to_indices(article, sentiment)

        ids = {i.index_id for i in impacts}
        assert "NIFTY50" in ids
        assert "SENSEX" in ids

    def test_direct_mention_confidence(self, mapper: NewsImpactMapper) -> None:
        """Direct mentions get 0.9 confidence."""
        article = _make_article(
            title="Nifty IT falls",
            mentioned_indices=["NIFTY IT"],
        )
        sentiment = _make_sentiment(adjusted_score=-0.4)
        impacts = mapper.map_to_indices(article, sentiment)

        nifty_it = [i for i in impacts if i.index_id == "NIFTY_IT"]
        assert len(nifty_it) >= 1
        assert nifty_it[0].confidence >= 0.9


# ---------------------------------------------------------------------------
# Strategy 2: Company-to-index mapping
# ---------------------------------------------------------------------------

class TestCompanyMapping:
    def test_reliance_maps_to_nifty_sensex(self, mapper: NewsImpactMapper) -> None:
        """Reliance news maps to NIFTY50 and SENSEX."""
        article = _make_article(
            title="Reliance Industries Q3 profit rises 20%",
            clean_text="Reliance Industries Q3 profit rises 20% driven by Jio",
            mentioned_companies=["Reliance Industries"],
            event_type="EARNINGS",
        )
        sentiment = _make_sentiment(adjusted_score=0.5)
        impacts = mapper.map_to_indices(article, sentiment)

        ids = {i.index_id for i in impacts}
        assert "NIFTY50" in ids
        assert "SENSEX" in ids

    def test_hdfc_bank_maps_to_banknifty(self, mapper: NewsImpactMapper) -> None:
        """HDFC Bank news maps to BANKNIFTY with high confidence."""
        article = _make_article(
            title="HDFC Bank Q3 profit rises 20%",
            clean_text="HDFC Bank Q3 profit rises 20% on strong NII growth",
            mentioned_companies=["HDFC Bank"],
            event_type="EARNINGS",
        )
        sentiment = _make_sentiment(adjusted_score=0.5)
        impacts = mapper.map_to_indices(article, sentiment)

        banknifty = [i for i in impacts if i.index_id == "BANKNIFTY"]
        assert len(banknifty) >= 1
        assert banknifty[0].confidence >= 0.75
        assert banknifty[0].company_name == "HDFC Bank"
        # HDFC Bank is ~28% of BANKNIFTY
        assert banknifty[0].estimated_index_weight is not None
        assert banknifty[0].estimated_index_weight > 5.0

    def test_tcs_maps_to_nifty_it(self, mapper: NewsImpactMapper) -> None:
        """TCS news maps to NIFTY_IT."""
        article = _make_article(
            title="TCS wins mega deal",
            mentioned_companies=["TCS"],
            event_type="CORPORATE",
        )
        sentiment = _make_sentiment(adjusted_score=0.4)
        impacts = mapper.map_to_indices(article, sentiment)

        nifty_it = [i for i in impacts if i.index_id == "NIFTY_IT"]
        assert len(nifty_it) >= 1

    def test_company_not_in_config(self, mapper: NewsImpactMapper) -> None:
        """Unknown company doesn't crash; other strategies still work."""
        article = _make_article(
            title="Unknown Corp Q3 results strong",
            clean_text="Unknown Corp Q3 results strong banking sector benefits",
            mentioned_companies=["Unknown Corp"],
        )
        sentiment = _make_sentiment(adjusted_score=0.3)
        # Should not raise
        impacts = mapper.map_to_indices(article, sentiment)
        # May still get keyword hits from "banking sector"
        assert isinstance(impacts, list)

    def test_major_company_weight(self, mapper: NewsImpactMapper) -> None:
        """Major companies (weight > 5%) get higher confidence."""
        article = _make_article(
            title="HDFC Bank NPA issue",
            mentioned_companies=["HDFC Bank"],
        )
        sentiment = _make_sentiment(adjusted_score=-0.3)
        impacts = mapper.map_to_indices(article, sentiment)

        banknifty = [i for i in impacts if i.index_id == "BANKNIFTY"]
        assert len(banknifty) >= 1
        # HDFC Bank weight > 5% → confidence 0.8
        assert banknifty[0].confidence >= 0.8


# ---------------------------------------------------------------------------
# Strategy 3: Keyword mapping
# ---------------------------------------------------------------------------

class TestKeywordMapping:
    def test_rbi_rate_cut_maps_to_banknifty(self, mapper: NewsImpactMapper) -> None:
        """RBI keywords map to banking indices."""
        article = _make_article(
            title="RBI announces rate cut",
            clean_text="RBI announces rate cut of 25 basis points in repo rate",
            event_type="POLICY",
        )
        sentiment = _make_sentiment(adjusted_score=0.6)
        impacts = mapper.map_to_indices(article, sentiment)

        ids = {i.index_id for i in impacts}
        assert "BANKNIFTY" in ids
        assert "NIFTY50" in ids

    def test_crude_oil_maps_to_energy(self, mapper: NewsImpactMapper) -> None:
        """Crude oil keywords map to energy indices."""
        article = _make_article(
            title="Crude oil surges above $90",
            clean_text="Brent crude oil surges above $90 on opec supply cut",
        )
        sentiment = _make_sentiment(adjusted_score=-0.3)
        impacts = mapper.map_to_indices(article, sentiment)

        ids = {i.index_id for i in impacts}
        assert "NIFTY_ENERGY" in ids or "NIFTY_OIL_GAS" in ids

    def test_keyword_confidence_moderate(self, mapper: NewsImpactMapper) -> None:
        """Keyword-only mappings get moderate confidence (0.5-0.6)."""
        article = _make_article(
            title="Steel prices rise",
            clean_text="Steel price surges on iron ore demand from china pmi data",
        )
        sentiment = _make_sentiment(adjusted_score=0.3)
        impacts = mapper.map_to_indices(article, sentiment)

        # All keyword-only impacts should have moderate confidence
        keyword_only = [i for i in impacts if i.mapping_method == "KEYWORD_MAPPING"]
        for imp in keyword_only:
            assert 0.3 <= imp.confidence <= 0.65


# ---------------------------------------------------------------------------
# Multi-method confidence boost
# ---------------------------------------------------------------------------

class TestMultiMethod:
    def test_multi_method_boost(self, mapper: NewsImpactMapper) -> None:
        """Article with both direct mention and company gets boosted confidence."""
        article = _make_article(
            title="HDFC Bank drives Bank Nifty rally",
            clean_text="HDFC Bank drives Bank Nifty rally as banking stocks surge",
            mentioned_companies=["HDFC Bank"],
            mentioned_indices=["NIFTY BANK"],
        )
        sentiment = _make_sentiment(adjusted_score=0.6)
        impacts = mapper.map_to_indices(article, sentiment)

        banknifty = [i for i in impacts if i.index_id == "BANKNIFTY"]
        assert len(banknifty) == 1
        # Direct (0.9) + company → boost to min(0.9+0.1, 0.95) = 0.95
        assert banknifty[0].confidence >= 0.95
        assert banknifty[0].mapping_method == "MULTI_METHOD"

    def test_company_and_keyword_boost(self, mapper: NewsImpactMapper) -> None:
        """Company + keyword should boost confidence."""
        article = _make_article(
            title="HDFC Bank benefits from RBI rate cut",
            clean_text="HDFC Bank benefits from RBI rate cut in repo rate decision",
            mentioned_companies=["HDFC Bank"],
        )
        sentiment = _make_sentiment(adjusted_score=0.5)
        impacts = mapper.map_to_indices(article, sentiment)

        banknifty = [i for i in impacts if i.index_id == "BANKNIFTY"]
        assert len(banknifty) >= 1
        # Company (0.8) + keyword → boost → > 0.8
        assert banknifty[0].confidence > 0.8


# ---------------------------------------------------------------------------
# Directional impact
# ---------------------------------------------------------------------------

class TestDirectionalImpact:
    def test_positive_sentiment_positive_impact(self, mapper: NewsImpactMapper) -> None:
        """Bullish article → positive directional impact."""
        article = _make_article(
            title="Nifty hits record high",
            mentioned_indices=["NIFTY 50"],
        )
        sentiment = _make_sentiment(adjusted_score=0.7)
        impacts = mapper.map_to_indices(article, sentiment)

        nifty = [i for i in impacts if i.index_id == "NIFTY50"]
        assert len(nifty) >= 1
        assert nifty[0].directional_impact > 0

    def test_negative_sentiment_negative_impact(self, mapper: NewsImpactMapper) -> None:
        """Bearish article → negative directional impact."""
        article = _make_article(
            title="Sensex crashes 1000 points",
            mentioned_indices=["BSE SENSEX"],
        )
        sentiment = _make_sentiment(adjusted_score=-0.8, label="VERY_BEARISH")
        impacts = mapper.map_to_indices(article, sentiment)

        sensex = [i for i in impacts if i.index_id == "SENSEX"]
        assert len(sensex) >= 1
        assert sensex[0].directional_impact < 0


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

class TestSeverityClassification:
    def test_critical_policy_event(self, mapper: NewsImpactMapper) -> None:
        """POLICY event + high credibility + strong sentiment → CRITICAL."""
        article = _make_article(
            title="RBI surprises with 50bps rate cut",
            clean_text="RBI surprises with 50bps rate cut in emergency meeting",
            event_type="POLICY",
            source_credibility=0.9,
        )
        sentiment = _make_sentiment(adjusted_score=0.7)
        impacts = mapper.map_to_indices(article, sentiment)
        severity = mapper.classify_impact_severity(article, sentiment, impacts)
        assert severity == "CRITICAL"

    def test_critical_global_crisis(self, mapper: NewsImpactMapper) -> None:
        """GLOBAL event with crisis keywords → CRITICAL."""
        article = _make_article(
            title="Global markets crash on war fears",
            clean_text="Global markets crash on war fears as geopolitical tensions escalate",
            event_type="GLOBAL",
            source_credibility=0.9,
        )
        sentiment = _make_sentiment(adjusted_score=-0.8)
        impacts = mapper.map_to_indices(article, sentiment)
        severity = mapper.classify_impact_severity(article, sentiment, impacts)
        assert severity == "CRITICAL"

    def test_critical_high_weight_companies(self, mapper: NewsImpactMapper) -> None:
        """Multiple high-weight companies (>10% combined) → CRITICAL."""
        article = _make_article(
            title="Reliance and HDFC Bank results shock market",
            clean_text="Reliance and HDFC Bank results shock market with surprise misses",
            mentioned_companies=["Reliance Industries", "HDFC Bank"],
            event_type="EARNINGS",
        )
        sentiment = _make_sentiment(adjusted_score=-0.6)
        impacts = mapper.map_to_indices(article, sentiment)
        severity = mapper.classify_impact_severity(article, sentiment, impacts)
        # Reliance ~10.5% + HDFC ~8.5% = 19% combined in NIFTY50
        assert severity == "CRITICAL"

    def test_high_earnings_event(self, mapper: NewsImpactMapper) -> None:
        """EARNINGS event with strong sentiment → HIGH."""
        article = _make_article(
            title="Infosys Q3 results beat estimates",
            mentioned_companies=["Infosys"],
            event_type="EARNINGS",
            source_credibility=0.8,
        )
        sentiment = _make_sentiment(adjusted_score=0.5)
        impacts = mapper.map_to_indices(article, sentiment)
        severity = mapper.classify_impact_severity(article, sentiment, impacts)
        assert severity == "HIGH"

    def test_high_major_company(self, mapper: NewsImpactMapper) -> None:
        """Major company (weight > 5%) + moderate sentiment → HIGH."""
        article = _make_article(
            title="ICICI Bank provisions rise",
            mentioned_companies=["ICICI Bank"],
            source_credibility=0.8,
        )
        sentiment = _make_sentiment(adjusted_score=-0.4)
        impacts = mapper.map_to_indices(article, sentiment)
        severity = mapper.classify_impact_severity(article, sentiment, impacts)
        assert severity == "HIGH"

    def test_medium_corporate_event(self, mapper: NewsImpactMapper) -> None:
        """CORPORATE event with moderate sentiment → MEDIUM."""
        article = _make_article(
            title="Titan announces buyback",
            mentioned_companies=["Titan Company"],
            event_type="CORPORATE",
            source_credibility=0.7,
        )
        sentiment = _make_sentiment(adjusted_score=0.3)
        impacts = mapper.map_to_indices(article, sentiment)
        severity = mapper.classify_impact_severity(article, sentiment, impacts)
        assert severity == "MEDIUM"

    def test_low_weak_sentiment(self, mapper: NewsImpactMapper) -> None:
        """Weak sentiment (< 0.2) → LOW regardless of event type."""
        article = _make_article(
            title="Market remains flat",
            clean_text="Market remains flat in early trade with no clear direction",
            mentioned_indices=["NIFTY 50"],
            event_type="MARKET_MOVE",
            source_credibility=0.8,
        )
        sentiment = _make_sentiment(adjusted_score=0.1)
        impacts = mapper.map_to_indices(article, sentiment)
        severity = mapper.classify_impact_severity(article, sentiment, impacts)
        assert severity == "LOW"

    def test_low_credibility_source(self, mapper: NewsImpactMapper) -> None:
        """Low credibility source → LOW."""
        article = _make_article(
            title="Market to rally 10% next week",
            clean_text="Market to rally 10% next week according to predictions",
            mentioned_indices=["NIFTY 50"],
            source_credibility=0.3,
        )
        sentiment = _make_sentiment(adjusted_score=0.4)
        impacts = mapper.map_to_indices(article, sentiment)
        severity = mapper.classify_impact_severity(article, sentiment, impacts)
        assert severity == "LOW"

    def test_noise_no_mapping(self, mapper: NewsImpactMapper) -> None:
        """Article with no index mapping → NOISE."""
        article = _make_article(
            title="Local dog show winners announced",
            clean_text="Local dog show winners announced at the park yesterday afternoon",
        )
        sentiment = _make_sentiment(adjusted_score=0.1)
        impacts = mapper.map_to_indices(article, sentiment)
        severity = mapper.classify_impact_severity(article, sentiment, impacts)
        assert severity == "NOISE"

    def test_noise_very_low_sentiment(self, mapper: NewsImpactMapper) -> None:
        """Extremely low sentiment → NOISE."""
        article = _make_article(
            title="Market update",
            mentioned_indices=["NIFTY 50"],
        )
        sentiment = _make_sentiment(adjusted_score=0.02)
        impacts = mapper.map_to_indices(article, sentiment)
        severity = mapper.classify_impact_severity(article, sentiment, impacts)
        assert severity == "NOISE"

    def test_old_article_is_low(self, mapper: NewsImpactMapper) -> None:
        """Article older than 24 hours → always LOW."""
        article = _make_article(
            title="RBI rate cut yesterday",
            clean_text="RBI announced rate cut in an emergency meeting yesterday",
            event_type="POLICY",
            source_credibility=0.9,
            published_at=_now_ist() - timedelta(hours=30),
        )
        sentiment = _make_sentiment(adjusted_score=0.8)
        impacts = mapper.map_to_indices(article, sentiment)
        severity = mapper.classify_impact_severity(article, sentiment, impacts)
        assert severity == "LOW"


# ---------------------------------------------------------------------------
# map_and_classify (combined pipeline)
# ---------------------------------------------------------------------------

class TestMapAndClassify:
    def test_returns_mapped_article(self, mapper: NewsImpactMapper) -> None:
        article = _make_article(
            title="Bank Nifty rallies on HDFC Bank results",
            clean_text="Bank Nifty rallies on HDFC Bank results beating estimates",
            mentioned_companies=["HDFC Bank"],
            mentioned_indices=["NIFTY BANK"],
            event_type="EARNINGS",
            source_credibility=0.8,
        )
        sentiment = _make_sentiment(adjusted_score=0.6)
        result = mapper.map_and_classify(article, sentiment)

        assert isinstance(result, MappedArticle)
        assert result.article is article
        assert result.sentiment is sentiment
        assert len(result.index_impacts) > 0
        assert result.primary_index is not None
        assert result.impact_severity in _SEVERITY_ORDER

    def test_actionable_flag(self, mapper: NewsImpactMapper) -> None:
        """CRITICAL/HIGH severity with index impacts → is_actionable = True."""
        article = _make_article(
            title="RBI emergency rate cut",
            clean_text="RBI emergency rate cut as monetary policy changes drastically",
            event_type="POLICY",
            source_credibility=0.9,
        )
        sentiment = _make_sentiment(adjusted_score=0.7)
        result = mapper.map_and_classify(article, sentiment)

        assert result.impact_severity in ("CRITICAL", "HIGH")
        assert result.is_actionable is True

    def test_non_actionable_low(self, mapper: NewsImpactMapper) -> None:
        """LOW severity → not actionable."""
        article = _make_article(
            title="Market flat today",
            clean_text="Markets remained flat today with low volumes",
            mentioned_indices=["NIFTY 50"],
            source_credibility=0.8,
        )
        sentiment = _make_sentiment(adjusted_score=0.05)
        result = mapper.map_and_classify(article, sentiment)

        assert result.is_actionable is False

    def test_primary_index_is_highest_confidence(self, mapper: NewsImpactMapper) -> None:
        """Primary index should be the one with highest confidence."""
        article = _make_article(
            title="Bank Nifty surges",
            mentioned_indices=["NIFTY BANK"],
        )
        sentiment = _make_sentiment(adjusted_score=0.5)
        result = mapper.map_and_classify(article, sentiment)

        if result.index_impacts:
            assert result.primary_index == result.index_impacts[0].index_id


# ---------------------------------------------------------------------------
# Index-specific news feed
# ---------------------------------------------------------------------------

class TestIndexNewsFeed:
    def _make_mapped(
        self,
        mapper: NewsImpactMapper,
        title: str,
        index_ids: list[str],
        severity: str,
        directional: float = 0.5,
    ) -> MappedArticle:
        article = _make_article(title=title)
        sentiment = _make_sentiment(adjusted_score=directional)
        impacts = [
            IndexImpact(
                index_id=idx,
                relevance_score=0.8,
                mapping_method="DIRECT_MENTION",
                confidence=0.8,
                sentiment_score=directional,
                directional_impact=0.8 * directional,
            )
            for idx in index_ids
        ]
        return MappedArticle(
            article=article,
            sentiment=sentiment,
            index_impacts=impacts,
            impact_severity=severity,
            primary_index=index_ids[0] if index_ids else None,
            is_actionable=severity in ("CRITICAL", "HIGH"),
        )

    def test_filters_by_index(self, mapper: NewsImpactMapper) -> None:
        articles = [
            self._make_mapped(mapper, "BankNifty news", ["BANKNIFTY"], "HIGH"),
            self._make_mapped(mapper, "IT news", ["NIFTY_IT"], "MEDIUM"),
            self._make_mapped(mapper, "Both", ["BANKNIFTY", "NIFTY_IT"], "LOW"),
        ]
        feed = NewsImpactMapper.get_index_news_feed("BANKNIFTY", articles)
        assert len(feed) == 2
        titles = [m.article.title for m in feed]
        assert "BankNifty news" in titles
        assert "Both" in titles
        assert "IT news" not in titles

    def test_filters_by_severity(self, mapper: NewsImpactMapper) -> None:
        articles = [
            self._make_mapped(mapper, "Critical", ["NIFTY50"], "CRITICAL"),
            self._make_mapped(mapper, "High", ["NIFTY50"], "HIGH"),
            self._make_mapped(mapper, "Low", ["NIFTY50"], "LOW"),
            self._make_mapped(mapper, "Noise", ["NIFTY50"], "NOISE"),
        ]
        feed = NewsImpactMapper.get_index_news_feed(
            "NIFTY50", articles, min_severity="HIGH"
        )
        assert len(feed) == 2
        severities = [m.impact_severity for m in feed]
        assert "CRITICAL" in severities
        assert "HIGH" in severities
        assert "LOW" not in severities

    def test_sorted_by_severity_then_impact(self, mapper: NewsImpactMapper) -> None:
        articles = [
            self._make_mapped(mapper, "Low impact", ["NIFTY50"], "MEDIUM", 0.1),
            self._make_mapped(mapper, "Critical", ["NIFTY50"], "CRITICAL", 0.9),
            self._make_mapped(mapper, "High impact", ["NIFTY50"], "MEDIUM", 0.8),
        ]
        feed = NewsImpactMapper.get_index_news_feed("NIFTY50", articles)
        assert feed[0].article.title == "Critical"
        # Among MEDIUMs, higher directional impact first
        medium_feed = [m for m in feed if m.impact_severity == "MEDIUM"]
        assert medium_feed[0].article.title == "High impact"


# ---------------------------------------------------------------------------
# Market news summary
# ---------------------------------------------------------------------------

class TestMarketNewsSummary:
    def _make_mapped(
        self,
        severity: str,
        sentiment: float,
        event_type: str = "EARNINGS",
        index_ids: list[str] | None = None,
        credibility: float = 0.8,
    ) -> MappedArticle:
        index_ids = index_ids or ["NIFTY50"]
        article = _make_article(
            title=f"Article {severity}",
            event_type=event_type,
            source_credibility=credibility,
        )
        sent = _make_sentiment(adjusted_score=sentiment)
        impacts = [
            IndexImpact(
                index_id=idx,
                relevance_score=0.8,
                mapping_method="DIRECT_MENTION",
                confidence=0.8,
                sentiment_score=sentiment,
                directional_impact=0.8 * sentiment,
            )
            for idx in index_ids
        ]
        return MappedArticle(
            article=article,
            sentiment=sent,
            index_impacts=impacts,
            impact_severity=severity,
            primary_index=index_ids[0],
            is_actionable=severity in ("CRITICAL", "HIGH"),
        )

    def test_empty_articles(self) -> None:
        summary = NewsImpactMapper.get_market_news_summary([])
        assert summary.total_articles == 0
        assert summary.overall_sentiment_label == "NEUTRAL"

    def test_severity_counts(self) -> None:
        articles = [
            self._make_mapped("CRITICAL", 0.8),
            self._make_mapped("HIGH", 0.5),
            self._make_mapped("HIGH", 0.4),
            self._make_mapped("MEDIUM", 0.2),
            self._make_mapped("LOW", 0.05),
        ]
        summary = NewsImpactMapper.get_market_news_summary(articles)

        assert summary.total_articles == 5
        assert summary.by_severity["CRITICAL"] == 1
        assert summary.by_severity["HIGH"] == 2
        assert summary.by_severity["MEDIUM"] == 1
        assert summary.by_severity["LOW"] == 1

    def test_overall_bullish_sentiment(self) -> None:
        articles = [
            self._make_mapped("HIGH", 0.7),
            self._make_mapped("HIGH", 0.5),
            self._make_mapped("MEDIUM", 0.3),
        ]
        summary = NewsImpactMapper.get_market_news_summary(articles)
        assert summary.overall_market_sentiment > 0.1
        assert summary.overall_sentiment_label == "BULLISH"

    def test_overall_bearish_sentiment(self) -> None:
        articles = [
            self._make_mapped("HIGH", -0.7),
            self._make_mapped("HIGH", -0.5),
            self._make_mapped("MEDIUM", -0.3),
        ]
        summary = NewsImpactMapper.get_market_news_summary(articles)
        assert summary.overall_market_sentiment < -0.1
        assert summary.overall_sentiment_label == "BEARISH"

    def test_critical_alerts(self) -> None:
        articles = [
            self._make_mapped("CRITICAL", -0.8),
            self._make_mapped("HIGH", 0.5),
        ]
        summary = NewsImpactMapper.get_market_news_summary(articles)
        assert len(summary.critical_alerts) == 1
        assert "CRITICAL" in summary.critical_alerts[0]

    def test_most_impacted_indices(self) -> None:
        articles = [
            self._make_mapped("HIGH", 0.5, index_ids=["BANKNIFTY"]),
            self._make_mapped("HIGH", 0.5, index_ids=["BANKNIFTY"]),
            self._make_mapped("MEDIUM", 0.3, index_ids=["NIFTY50"]),
        ]
        summary = NewsImpactMapper.get_market_news_summary(articles)
        assert len(summary.most_impacted_indices) > 0
        # BANKNIFTY should be most impacted (2 articles)
        assert summary.most_impacted_indices[0][0] == "BANKNIFTY"

    def test_sentiment_distribution(self) -> None:
        articles = [
            self._make_mapped("HIGH", 0.5),   # bullish
            self._make_mapped("HIGH", -0.3),   # bearish
            self._make_mapped("LOW", 0.01),    # neutral
        ]
        summary = NewsImpactMapper.get_market_news_summary(articles)
        assert summary.sentiment_distribution["bullish"] == 1
        assert summary.sentiment_distribution["bearish"] == 1
        assert summary.sentiment_distribution["neutral"] == 1

    def test_dominant_event_type(self) -> None:
        articles = [
            self._make_mapped("HIGH", 0.5, event_type="EARNINGS"),
            self._make_mapped("HIGH", 0.4, event_type="EARNINGS"),
            self._make_mapped("MEDIUM", 0.3, event_type="POLICY"),
        ]
        summary = NewsImpactMapper.get_market_news_summary(articles)
        assert summary.dominant_event_type == "EARNINGS"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_mapping_found(self, mapper: NewsImpactMapper) -> None:
        """Article with no companies, sectors, or indices → empty impacts."""
        article = _make_article(
            title="Best pasta recipes for dinner",
            clean_text="Best pasta recipes for dinner with fresh basil and cheese",
        )
        sentiment = _make_sentiment(adjusted_score=0.0)
        impacts = mapper.map_to_indices(article, sentiment)
        assert len(impacts) == 0

    def test_broad_news_penalty(self, mapper: NewsImpactMapper) -> None:
        """Article affecting 10+ indices gets confidence reduced."""
        # Use text that triggers many keyword categories at once
        article = _make_article(
            title="Budget impacts all sectors",
            clean_text=(
                "union budget fiscal deficit disinvestment gst collection "
                "rbi repo rate banking sector credit growth "
                "pharma drug approval healthcare "
                "crude oil brent opec energy sector "
                "steel price iron ore metal "
                "auto sales vehicle sales ev electric vehicle "
                "fmcg rural consumption "
                "real estate housing market "
                "infrastructure road construction capex "
                "sensex crash nifty fall market crash "
                "fed rate us interest rate global recession "
                "telecom airtel 5g rollout "
            ),
        )
        sentiment = _make_sentiment(adjusted_score=0.5)
        impacts = mapper.map_to_indices(article, sentiment)

        if len(impacts) > 10:
            # All impacts should have reduced confidence
            for imp in impacts:
                assert imp.confidence <= 0.8  # original max ~0.6 for keywords

    def test_article_with_empty_companies_list(self, mapper: NewsImpactMapper) -> None:
        """Company in config with empty indices list doesn't crash."""
        article = _make_article(
            title="Adani Power plant news",
            mentioned_companies=["Adani Power"],
        )
        sentiment = _make_sentiment(adjusted_score=0.3)
        # Adani Power has empty indices list in config
        impacts = mapper.map_to_indices(article, sentiment)
        # Should still work — may get keyword hits but no company mapping
        assert isinstance(impacts, list)

    def test_stale_article_during_market(self, mapper: NewsImpactMapper) -> None:
        """Article > 6 hours old with moderate content classified as LOW."""
        article = _make_article(
            title="Dabur launches new product",
            clean_text="Dabur launches new product in the fmcg consumer segment",
            mentioned_companies=["Dabur India"],
            event_type="CORPORATE",
            source_credibility=0.7,
            published_at=_now_ist() - timedelta(hours=8),
        )
        sentiment = _make_sentiment(adjusted_score=0.3)
        impacts = mapper.map_to_indices(article, sentiment)
        severity = mapper.classify_impact_severity(article, sentiment, impacts)
        assert severity == "LOW"

    def test_confidence_capped_at_095(self, mapper: NewsImpactMapper) -> None:
        """Even with multi-method boost, confidence never exceeds 0.95."""
        article = _make_article(
            title="HDFC Bank drives Bank Nifty rally on RBI rate cut",
            clean_text="HDFC Bank drives Bank Nifty rally on RBI rate cut news",
            mentioned_companies=["HDFC Bank"],
            mentioned_indices=["NIFTY BANK"],
        )
        sentiment = _make_sentiment(adjusted_score=0.6)
        impacts = mapper.map_to_indices(article, sentiment)

        for imp in impacts:
            assert imp.confidence <= 0.95
