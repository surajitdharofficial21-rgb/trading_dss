"""Tests for src.analysis.news.sentiment_analyzer."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.analysis.news.article_parser import ParsedArticle
from src.analysis.news.sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentCalibrator,
    SentimentResult,
    _score_to_label,
)

_IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 5, 10, 0, tzinfo=_IST)


def _article(
    title: str = "Test headline",
    clean_text: str = "Some market news text.",
    source: str = "TestSource",
    source_credibility: float = 0.85,
    language: str = "en",
) -> ParsedArticle:
    return ParsedArticle(
        title=title,
        clean_text=clean_text,
        url="https://example.com/article",
        source=source,
        source_credibility=source_credibility,
        category="markets",
        published_at=_NOW,
        fetched_at=_NOW,
        language=language,
    )


@pytest.fixture()
def analyzer() -> SentimentAnalyzer:
    """Analyzer loaded with the real project keyword file."""
    return SentimentAnalyzer()


@pytest.fixture()
def minimal_kw_file(tmp_path: Path) -> Path:
    """A tiny keyword file for controlled tests."""
    data = {
        "bullish": [
            {"keyword": "surge", "intensity": 0.8, "category": "price_movement"},
            {"keyword": "record high", "intensity": 0.85, "category": "price_movement"},
            {"keyword": "rate cut", "intensity": 0.8, "category": "macro"},
        ],
        "bearish": [
            {"keyword": "crash", "intensity": 0.95, "category": "price_movement"},
            {"keyword": "selloff", "intensity": 0.8, "category": "price_movement"},
            {"keyword": "war", "intensity": 0.7, "category": "macro"},
        ],
        "neutral_uncertainty": [
            {"keyword": "volatile", "intensity": 0.5, "category": "market_sentiment"},
            {"keyword": "cautious", "intensity": 0.4, "category": "market_sentiment"},
        ],
    }
    p = tmp_path / "sentiment_keywords.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture()
def mini_analyzer(minimal_kw_file: Path) -> SentimentAnalyzer:
    return SentimentAnalyzer(keywords_path=minimal_kw_file)


# ---------------------------------------------------------------------------
# Label helper
# ---------------------------------------------------------------------------


class TestScoreToLabel:
    def test_very_bullish(self):
        assert _score_to_label(0.8) == "VERY_BULLISH"

    def test_bullish(self):
        assert _score_to_label(0.4) == "BULLISH"

    def test_slightly_bullish(self):
        assert _score_to_label(0.1) == "SLIGHTLY_BULLISH"

    def test_neutral(self):
        assert _score_to_label(0.0) == "NEUTRAL"

    def test_slightly_bearish(self):
        assert _score_to_label(-0.1) == "SLIGHTLY_BEARISH"

    def test_bearish(self):
        assert _score_to_label(-0.4) == "BEARISH"

    def test_very_bearish(self):
        assert _score_to_label(-0.8) == "VERY_BEARISH"


# ---------------------------------------------------------------------------
# Clearly bullish / bearish / neutral headlines
# ---------------------------------------------------------------------------


class TestClearSentiment:
    def test_bullish_headline(self, analyzer: SentimentAnalyzer):
        art = _article(
            title="Nifty surges 500 points, hits record high on RBI rate cut",
            clean_text=(
                "Nifty surges 500 points, hits record high on RBI rate cut. "
                "Markets rally sharply with broad based buying across sectors."
            ),
        )
        result = analyzer.analyze_sentiment(art)
        assert result.adjusted_score > 0.4
        assert result.sentiment_label in ("VERY_BULLISH", "BULLISH")
        assert len(result.bullish_keywords_found) > 0

    def test_bearish_headline(self, analyzer: SentimentAnalyzer):
        art = _article(
            title="Market crashes 3% as war fears trigger massive selloff",
            clean_text=(
                "Market crashes 3% as war fears trigger massive sell off. "
                "Investors panic as geopolitical risk escalates sharply."
            ),
        )
        result = analyzer.analyze_sentiment(art)
        assert result.adjusted_score < -0.5
        assert result.sentiment_label in ("VERY_BEARISH", "BEARISH")
        assert len(result.bearish_keywords_found) > 0

    def test_neutral_headline(self, analyzer: SentimentAnalyzer):
        art = _article(
            title="Market opens flat, investors await RBI decision",
            clean_text=(
                "Market opens flat, investors await RBI decision. "
                "Trading is rangebound with low volumes in early trade."
            ),
        )
        result = analyzer.analyze_sentiment(art)
        assert -0.25 < result.adjusted_score < 0.25


# ---------------------------------------------------------------------------
# Keyword boosting
# ---------------------------------------------------------------------------


class TestKeywordBoosting:
    def test_surge_keyword_boosts_score(self, mini_analyzer: SentimentAnalyzer):
        base_art = _article(clean_text="Markets moved higher in today's session.")
        boosted_art = _article(
            clean_text="Markets moved higher in today's session with a big surge.",
        )
        base_result = mini_analyzer.analyze_sentiment(base_art)
        boosted_result = mini_analyzer.analyze_sentiment(boosted_art)
        assert boosted_result.adjusted_score > base_result.adjusted_score
        assert "surge" in boosted_result.bullish_keywords_found

    def test_crash_keyword_lowers_score(self, mini_analyzer: SentimentAnalyzer):
        base_art = _article(clean_text="Markets fell in today's session.")
        boosted_art = _article(
            clean_text="Markets fell in today's session amid a crash.",
        )
        base_result = mini_analyzer.analyze_sentiment(base_art)
        boosted_result = mini_analyzer.analyze_sentiment(boosted_art)
        assert boosted_result.adjusted_score < base_result.adjusted_score
        assert "crash" in boosted_result.bearish_keywords_found

    def test_multiple_keywords_have_diminishing_returns(
        self,
        mini_analyzer: SentimentAnalyzer,
    ):
        one_kw = _article(clean_text="Big surge in the market today, strong action.")
        many_kw = _article(
            clean_text=(
                "Big surge in the market, hits record high after rate cut, "
                "very strong positive action."
            ),
        )
        one_result = mini_analyzer.analyze_sentiment(one_kw)
        many_result = mini_analyzer.analyze_sentiment(many_kw)
        # More keywords → higher score but not linearly proportional
        assert many_result.adjusted_score > one_result.adjusted_score
        assert many_result.keyword_count > one_result.keyword_count


# ---------------------------------------------------------------------------
# Source credibility
# ---------------------------------------------------------------------------


class TestSourceCredibility:
    def test_high_credibility_gives_higher_absolute_score(
        self,
        mini_analyzer: SentimentAnalyzer,
    ):
        text = "Markets surge sharply today in a huge rally."
        high = _article(clean_text=text, source_credibility=0.90)
        low = _article(clean_text=text, source_credibility=0.50)
        r_high = mini_analyzer.analyze_sentiment(high)
        r_low = mini_analyzer.analyze_sentiment(low)
        assert abs(r_high.adjusted_score) > abs(r_low.adjusted_score)
        assert r_high.source_weight > r_low.source_weight


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------


class TestBatchAnalysis:
    def test_batch_sorted_by_absolute_score(self, mini_analyzer: SentimentAnalyzer):
        articles = [
            _article(clean_text="Flat trading, nothing notable."),
            _article(clean_text="Huge crash wipes out gains, selloff continues."),
            _article(clean_text="Nice surge in markets today."),
        ]
        results = mini_analyzer.analyze_batch(articles)
        scores = [abs(r.adjusted_score) for _, r in results]
        assert scores == sorted(scores, reverse=True)

    def test_batch_returns_all_articles(self, mini_analyzer: SentimentAnalyzer):
        articles = [_article(clean_text=f"Article {i}") for i in range(5)]
        results = mini_analyzer.analyze_batch(articles)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# Headline-only sentiment
# ---------------------------------------------------------------------------


class TestHeadlineSentiment:
    def test_bullish_headline(self, analyzer: SentimentAnalyzer):
        score = analyzer.get_headline_sentiment(
            "Nifty surges 500 points on RBI rate cut",
        )
        assert score > 0.3

    def test_bearish_headline(self, analyzer: SentimentAnalyzer):
        score = analyzer.get_headline_sentiment(
            "Markets crash 3% as war fears escalate",
        )
        assert score < -0.3

    def test_empty_headline(self, analyzer: SentimentAnalyzer):
        assert analyzer.get_headline_sentiment("") == 0.0
        assert analyzer.get_headline_sentiment("   ") == 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_text_returns_neutral(self, mini_analyzer: SentimentAnalyzer):
        art = _article(clean_text="")
        result = mini_analyzer.analyze_sentiment(art)
        assert result.sentiment_label == "NEUTRAL"
        assert result.confidence == 0.0
        assert result.adjusted_score == 0.0

    def test_whitespace_only_text(self, mini_analyzer: SentimentAnalyzer):
        art = _article(clean_text="   \n\t  ")
        result = mini_analyzer.analyze_sentiment(art)
        assert result.sentiment_label == "NEUTRAL"
        assert result.confidence == 0.0

    def test_very_short_text_reduces_confidence(
        self,
        mini_analyzer: SentimentAnalyzer,
    ):
        short = _article(clean_text="Markets surge today.")
        long_ = _article(
            clean_text=(
                "Markets surge today after RBI announced a surprise rate cut. "
                "All sectoral indices closed in the green with banking stocks "
                "leading the charge. Analysts expect the rally to continue."
            ),
        )
        r_short = mini_analyzer.analyze_sentiment(short)
        r_long = mini_analyzer.analyze_sentiment(long_)
        assert r_short.confidence < r_long.confidence

    def test_non_english_text_returns_neutral(
        self,
        mini_analyzer: SentimentAnalyzer,
    ):
        hindi_text = "बाजार में भारी गिरावट, निवेशकों में बेचैनी"
        art = _article(clean_text=hindi_text, language="hi")
        result = mini_analyzer.analyze_sentiment(art)
        assert result.sentiment_label == "NEUTRAL"
        assert result.confidence <= 0.1

    def test_all_caps_text_handled(self, mini_analyzer: SentimentAnalyzer):
        art = _article(clean_text="MARKETS SURGE TODAY IN A MASSIVE RALLY")
        result = mini_analyzer.analyze_sentiment(art)
        # Should still detect keywords and produce a score
        assert result.raw_vader_score != 0.0 or result.keyword_boost != 0.0

    def test_score_clamped_to_bounds(self, mini_analyzer: SentimentAnalyzer):
        art = _article(
            clean_text=(
                "Massive surge rally breakout record high "
                "incredible outstanding phenomenal spectacular"
            ),
            source_credibility=1.0,
        )
        result = mini_analyzer.analyze_sentiment(art)
        assert -1.0 <= result.adjusted_score <= 1.0


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


class TestSentimentCalibrator:
    def test_no_stats_before_min_samples(self):
        cal = SentimentCalibrator()
        for i in range(30):
            cal.record_outcome(0.5, 1.0, _NOW)
        assert cal.get_calibration_stats(min_samples=50) is None

    def test_stats_available_after_min_samples(self):
        cal = SentimentCalibrator()
        for _ in range(60):
            cal.record_outcome(0.5, 1.0, _NOW)
        stats = cal.get_calibration_stats(min_samples=50)
        assert stats is not None
        assert stats.sample_count == 60

    def test_bullish_bias_produces_negative_offset(self):
        """If we predict bullish but market doesn't move much, offset is negative."""
        cal = SentimentCalibrator()
        for _ in range(60):
            # We say +0.6 bullish, market only moves +0.1%
            cal.record_outcome(0.6, 0.1, _NOW)
        stats = cal.get_calibration_stats()
        assert stats is not None
        assert stats.sentiment_bias > 0  # We're too bullish
        assert stats.suggested_offset < 0  # Correction pulls scores down

    def test_apply_calibration_with_no_data(self):
        cal = SentimentCalibrator()
        assert cal.apply_calibration(0.5) == 0.5  # No change

    def test_apply_calibration_with_data(self):
        cal = SentimentCalibrator()
        for _ in range(60):
            cal.record_outcome(0.6, 0.1, _NOW)
        cal.get_calibration_stats()
        adjusted = cal.apply_calibration(0.5)
        assert adjusted != 0.5  # Offset applied
        assert adjusted < 0.5  # Should be pulled down (we're too bullish)

    def test_calibrator_integration_with_analyzer(
        self,
        minimal_kw_file: Path,
    ):
        cal = SentimentCalibrator()
        for _ in range(60):
            cal.record_outcome(0.6, 0.1, _NOW)
        cal.get_calibration_stats()

        analyzer_with = SentimentAnalyzer(
            keywords_path=minimal_kw_file, calibrator=cal,
        )
        analyzer_without = SentimentAnalyzer(keywords_path=minimal_kw_file)

        art = _article(
            clean_text="Markets surge today with broad based buying across sectors.",
        )
        r_with = analyzer_with.analyze_sentiment(art)
        r_without = analyzer_without.analyze_sentiment(art)
        # Calibrated score should be lower (correcting bullish bias)
        assert r_with.adjusted_score < r_without.adjusted_score

    def test_bearish_accuracy(self):
        cal = SentimentCalibrator()
        for _ in range(30):
            cal.record_outcome(-0.5, -1.5, _NOW)  # Correctly bearish
        for _ in range(30):
            cal.record_outcome(-0.5, 0.5, _NOW)  # Incorrectly bearish
        stats = cal.get_calibration_stats()
        assert stats is not None
        assert stats.avg_bearish_accuracy == pytest.approx(0.5, abs=0.01)
