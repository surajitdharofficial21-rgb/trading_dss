"""
News ingestion, keyword routing, and sentiment scoring.

Fetches RSS feeds, scores each article with VADER + custom keyword weights,
and maps articles to affected indices using news_mappings.json.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """
    A single news article with sentiment scores and index mappings.

    Attributes
    ----------
    title:
        Article headline.
    summary:
        Short description or first paragraph.
    url:
        Link to the full article.
    source:
        Feed source name.
    published_at:
        Publication timestamp (Unix epoch).
    vader_compound:
        Raw VADER compound score (−1 to +1).
    adjusted_score:
        Score after applying domain keyword boosts.
    affected_indices:
        List of index IDs this article is relevant to.
    credibility_score:
        Source credibility weight (0–1).
    """

    title: str
    summary: str
    url: str
    source: str
    published_at: float
    vader_compound: float
    adjusted_score: float
    affected_indices: list[str] = field(default_factory=list)
    credibility_score: float = 0.7


class NewsEngine:
    """
    Fetches RSS feeds, scores articles, and routes them to indices.

    Parameters
    ----------
    feeds_path:
        Path to ``rss_feeds.json``.
    mappings_path:
        Path to ``news_mappings.json``.
    keywords_path:
        Path to ``sentiment_keywords.json``.
    """

    def __init__(
        self,
        feeds_path: Optional[Path] = None,
        mappings_path: Optional[Path] = None,
        keywords_path: Optional[Path] = None,
    ) -> None:
        self._feeds_path = feeds_path or settings.rss_feeds_path
        self._mappings_path = mappings_path or settings.news_mappings_path
        self._keywords_path = keywords_path or settings.sentiment_keywords_path

        self._vader = SentimentIntensityAnalyzer()
        self._feeds: list[dict] = self._load_json(self._feeds_path)
        self._mappings: dict = self._load_json(self._mappings_path)
        self._keywords: dict = self._load_json(self._keywords_path)

        # Build fast-lookup sets from keywords
        self._bullish: dict[str, float] = {
            kw["keyword"].lower(): kw["intensity"]
            for kw in self._keywords.get("bullish", [])
        }
        self._bearish: dict[str, float] = {
            kw["keyword"].lower(): kw["intensity"]
            for kw in self._keywords.get("bearish", [])
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _load_json(path: Path) -> dict | list:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to load %s: %s", path, exc)
            return {}

    def _adjust_score(self, text: str, base: float) -> float:
        """Apply domain keyword boosts on top of the VADER base score."""
        lower = text.lower()
        boost = 0.0
        for kw, intensity in self._bullish.items():
            if kw in lower:
                boost += intensity * 0.1
        for kw, intensity in self._bearish.items():
            if kw in lower:
                boost -= intensity * 0.1
        return max(-1.0, min(1.0, base + boost))

    def _map_to_indices(self, text: str) -> list[str]:
        """Return index IDs affected by *text* based on news_mappings.json."""
        lower = text.lower()
        matched: set[str] = set()
        for _category, mapping in self._mappings.items():
            if isinstance(mapping, dict) and "keywords" in mapping:
                for kw in mapping["keywords"]:
                    if kw.lower() in lower:
                        matched.update(mapping.get("affected_indices", []))
                        break
        return sorted(matched)

    # ── Public interface ──────────────────────────────────────────────────────

    def fetch_all(self, max_age_seconds: float = 3600.0) -> list[NewsArticle]:
        """
        Fetch and parse all active RSS feeds.

        Parameters
        ----------
        max_age_seconds:
            Ignore articles older than this many seconds (default 1 hour).

        Returns
        -------
        list[NewsArticle]:
            Scored and mapped articles, sorted newest-first.
        """
        active_feeds = [f for f in self._feeds if f.get("is_active", True)]
        articles: list[NewsArticle] = []
        cutoff = time.time() - max_age_seconds

        for feed_cfg in active_feeds:
            url = feed_cfg.get("url", "")
            if not url:
                continue
            try:
                parsed = feedparser.parse(url)
                credibility = float(feed_cfg.get("credibility_score", 0.7))
                source = feed_cfg.get("source_name", "Unknown")

                for entry in parsed.entries:
                    pub_ts = time.mktime(entry.get("published_parsed") or time.gmtime())
                    if pub_ts < cutoff:
                        continue

                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    url_article = entry.get("link", "")
                    combined = f"{title} {summary}"

                    scores = self._vader.polarity_scores(combined)
                    vader_compound = scores["compound"]
                    adjusted = self._adjust_score(combined, vader_compound)
                    affected = self._map_to_indices(combined)

                    articles.append(NewsArticle(
                        title=title,
                        summary=summary,
                        url=url_article,
                        source=source,
                        published_at=pub_ts,
                        vader_compound=vader_compound,
                        adjusted_score=adjusted,
                        affected_indices=affected,
                        credibility_score=credibility,
                    ))

            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to fetch feed %s: %s", url, exc)

        articles.sort(key=lambda a: a.published_at, reverse=True)
        logger.info("Fetched %d articles from %d feeds", len(articles), len(active_feeds))
        return articles

    def aggregate_sentiment(
        self,
        articles: list[NewsArticle],
        index_id: str,
    ) -> float:
        """
        Compute a single weighted sentiment score for *index_id*.

        Parameters
        ----------
        articles:
            List of recently fetched articles (from :meth:`fetch_all`).
        index_id:
            Target index ID to aggregate sentiment for.

        Returns
        -------
        float:
            Weighted average sentiment score in [−1, 1]. ``0.0`` if no
            relevant articles found.
        """
        relevant = [
            a for a in articles
            if index_id in a.affected_indices
        ]
        if not relevant:
            return 0.0

        total_weight = sum(a.credibility_score for a in relevant)
        if total_weight == 0.0:
            return 0.0

        weighted_score = sum(
            a.adjusted_score * a.credibility_score for a in relevant
        ) / total_weight
        return round(weighted_score, 4)
