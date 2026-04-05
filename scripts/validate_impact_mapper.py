#!/usr/bin/env python3
"""
Validate the NewsImpactMapper end-to-end.

Uses ArticleParser + SentimentAnalyzer + NewsImpactMapper with synthetic
RawArticles that cover a realistic spread of market scenarios.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.news.rss_fetcher import RawArticle
from src.analysis.news.article_parser import ArticleParser
from src.analysis.news.sentiment_analyzer import SentimentAnalyzer
from src.analysis.news.impact_mapper import NewsImpactMapper

_IST = ZoneInfo("Asia/Kolkata")


def _now() -> datetime:
    return datetime.now(tz=_IST)


# ── Synthetic articles covering diverse scenarios ─────────────────────────────

SAMPLE_ARTICLES: list[dict] = [
    {
        "title": "RBI surprises with 50bps rate cut in emergency MPC meeting",
        "summary": "Reserve Bank of India cuts repo rate by 50 basis points to boost growth",
        "source": "economic_times",
        "credibility": 0.90,
        "category": "banking_finance",
    },
    {
        "title": "HDFC Bank Q3 profit jumps 25%, beats Street estimates",
        "summary": "HDFC Bank reports strong quarterly results with net interest income rising 20% year on year",
        "source": "moneycontrol",
        "credibility": 0.85,
        "category": "banking_finance",
    },
    {
        "title": "Reliance Industries and Infosys drag Nifty 50 lower",
        "summary": "Nifty 50 falls 1.5% as Reliance Industries and Infosys report weak guidance. Sensex also drops sharply.",
        "source": "livemint",
        "credibility": 0.88,
        "category": "general_market",
    },
    {
        "title": "Global markets crash on Russia-Ukraine war escalation fears",
        "summary": "Wall Street crashes overnight as geopolitical tensions rise. Dow Jones falls 3%. Global recession fears mount.",
        "source": "ndtv_profit",
        "credibility": 0.80,
        "category": "global_macro",
    },
    {
        "title": "TCS wins $2 billion mega deal from European bank",
        "summary": "Tata Consultancy Services bags largest ever deal for digital transformation of a major European bank",
        "source": "business_standard",
        "credibility": 0.88,
        "category": "it_sector",
    },
    {
        "title": "Crude oil surges above $95 as OPEC cuts output",
        "summary": "Brent crude oil hits $95 per barrel after OPEC announces surprise production cut. Energy stocks rally.",
        "source": "economic_times",
        "credibility": 0.90,
        "category": "energy_oil",
    },
    {
        "title": "FIIs sell Rs 5000 crore worth of Indian equities in single session",
        "summary": "Foreign institutional investors continue selling spree with massive outflows from Indian markets",
        "source": "moneycontrol",
        "credibility": 0.85,
        "category": "general_market",
    },
    {
        "title": "Tata Steel and JSW Steel surge on steel price hike",
        "summary": "Metal stocks rally as Tata Steel and JSW Steel announce price hikes amid rising iron ore demand",
        "source": "livemint",
        "credibility": 0.88,
        "category": "metals_mining",
    },
    {
        "title": "Bank Nifty hits all-time high on strong credit growth data",
        "summary": "Bank Nifty crosses 50000 for the first time as banking sector reports strong credit offtake numbers",
        "source": "moneycontrol",
        "credibility": 0.85,
        "category": "banking_finance",
    },
    {
        "title": "Markets flat as traders await quarterly results season",
        "summary": "Nifty 50 and Sensex trade in narrow range with low volumes ahead of earnings",
        "source": "ndtv_profit",
        "credibility": 0.80,
        "category": "general_market",
    },
    {
        "title": "Sun Pharma gets USFDA approval for new drug",
        "summary": "Sun Pharmaceutical receives FDA approval for its generic drug application in the US market",
        "source": "economic_times",
        "credibility": 0.90,
        "category": "pharma_healthcare",
    },
    {
        "title": "Maruti Suzuki reports record auto sales in March",
        "summary": "Maruti Suzuki passenger vehicle sales jump 15% year on year driven by new SUV models",
        "source": "business_standard",
        "credibility": 0.88,
        "category": "auto_sector",
    },
]


def build_raw_articles() -> list[RawArticle]:
    """Convert sample dicts into RawArticle objects."""
    now = _now()
    articles = []
    for i, data in enumerate(SAMPLE_ARTICLES):
        articles.append(RawArticle(
            title=data["title"],
            summary=data["summary"],
            url=f"https://example.com/article/{i}",
            source=data["source"],
            source_credibility=data["credibility"],
            category=data["category"],
            published_at=now - timedelta(minutes=i * 10),
            fetched_at=now,
        ))
    return articles


def main() -> None:
    print("=" * 78)
    print("  NewsImpactMapper End-to-End Validation")
    print("=" * 78)

    # ── Step 1: Parse ─────────────────────────────────────────────────────────
    parser = ArticleParser()
    analyzer = SentimentAnalyzer()
    mapper = NewsImpactMapper()

    raw_articles = build_raw_articles()
    print(f"\nInput: {len(raw_articles)} synthetic articles\n")

    # ── Step 2: Parse + Sentiment ─────────────────────────────────────────────
    results: list[tuple] = []
    for raw in raw_articles:
        parsed = parser.parse_article(raw)
        if parsed is None:
            continue
        sentiment = analyzer.analyze_sentiment(parsed)
        results.append((parsed, sentiment))

    print(f"Parsed & analyzed: {len(results)} articles\n")

    # ── Step 3: Map & classify ────────────────────────────────────────────────
    mapped_articles = []
    for article, sentiment in results:
        mapped = mapper.map_and_classify(article, sentiment)
        mapped_articles.append(mapped)

    # ── Show results ──────────────────────────────────────────────────────────
    print("-" * 78)
    for m in mapped_articles:
        print(f"\n[{m.impact_severity:>8}] {m.article.title[:70]}")
        print(f"  Sentiment: {m.sentiment.sentiment_label} ({m.sentiment.adjusted_score:+.3f})")
        print(f"  Actionable: {m.is_actionable}")
        for impact in m.index_impacts[:5]:
            print(
                f"  -> {impact.index_id}: relevance={impact.relevance_score:.2f}, "
                f"direction={impact.directional_impact:+.4f}, method={impact.mapping_method}"
            )
        if len(m.index_impacts) > 5:
            print(f"  ... and {len(m.index_impacts) - 5} more indices")

    # ── Market summary ────────────────────────────────────────────────────────
    summary = mapper.get_market_news_summary(mapped_articles)
    print("\n" + "=" * 78)
    print("  Market News Summary")
    print("=" * 78)
    print(f"Total: {summary.total_articles} articles")
    print(f"By severity: {summary.by_severity}")
    print(f"By event type: {summary.by_event_type}")
    print(
        f"Overall sentiment: {summary.overall_sentiment_label} "
        f"({summary.overall_market_sentiment:+.4f})"
    )
    print(f"Sentiment distribution: {summary.sentiment_distribution}")
    print(f"Dominant event type: {summary.dominant_event_type}")
    print(f"Most impacted indices:")
    for idx, score in summary.most_impacted_indices:
        print(f"  {idx}: {score:.4f}")
    if summary.critical_alerts:
        print(f"Critical alerts:")
        for alert in summary.critical_alerts:
            print(f"  !! {alert}")

    # ── Index-specific feed ───────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  Index-Specific Feeds")
    print("=" * 78)

    for index_id in ["BANKNIFTY", "NIFTY50", "NIFTY_IT", "NIFTY_METAL"]:
        feed = mapper.get_index_news_feed(index_id, mapped_articles, min_severity="MEDIUM")
        print(f"\n{index_id} relevant news: {len(feed)} articles")
        for m in feed[:3]:
            best_impact = next(
                (i for i in m.index_impacts if i.index_id == index_id), None
            )
            dir_str = f"{best_impact.directional_impact:+.4f}" if best_impact else "N/A"
            print(f"  [{m.impact_severity:>8}] {m.article.title[:55]}  (dir={dir_str})")

    # ── Validation checks ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  Validation Checks")
    print("=" * 78)

    checks_passed = 0
    checks_total = 0

    def check(name: str, condition: bool) -> None:
        nonlocal checks_passed, checks_total
        checks_total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            checks_passed += 1
        print(f"  [{status}] {name}")

    check("All articles mapped", len(mapped_articles) == len(results))
    check("At least one CRITICAL article", any(m.impact_severity == "CRITICAL" for m in mapped_articles))
    check("At least one HIGH article", any(m.impact_severity == "HIGH" for m in mapped_articles))
    check("At least one MEDIUM article", any(m.impact_severity == "MEDIUM" for m in mapped_articles))
    check("At least one actionable article", any(m.is_actionable for m in mapped_articles))
    check("No articles without severity", all(m.impact_severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "NOISE") for m in mapped_articles))
    check("Summary total matches", summary.total_articles == len(mapped_articles))
    check("Summary has impacted indices", len(summary.most_impacted_indices) > 0)
    check("BANKNIFTY feed has articles", len(mapper.get_index_news_feed("BANKNIFTY", mapped_articles)) > 0)
    check("NIFTY50 feed has articles", len(mapper.get_index_news_feed("NIFTY50", mapped_articles)) > 0)
    check("Confidence never exceeds 0.95", all(
        imp.confidence <= 0.95
        for m in mapped_articles
        for imp in m.index_impacts
    ))
    check("Directional impact sign matches sentiment", all(
        (imp.directional_impact >= 0) == (imp.sentiment_score >= 0)
        for m in mapped_articles
        for imp in m.index_impacts
        if imp.sentiment_score != 0.0
    ))

    print(f"\n  Result: {checks_passed}/{checks_total} checks passed")
    if checks_passed == checks_total:
        print("  All validations PASSED!")
    else:
        print("  Some validations FAILED — review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
