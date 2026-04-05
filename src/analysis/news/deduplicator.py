"""
Article deduplication using URL matching and Jaccard title similarity.

Same news stories appear on multiple feeds.  This module detects near-
duplicates cheaply (no heavy NLP) and keeps the most credible version.
"""

from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import TYPE_CHECKING

from src.analysis.news.article_parser import ParsedArticle

if TYPE_CHECKING:
    from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


def _title_words(title: str) -> set[str]:
    """Lowercase, strip punctuation, return word set."""
    cleaned = re.sub(r"[^\w\s]", "", title.lower())
    return set(cleaned.split())


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ArticleDeduplicator:
    """
    Detects duplicate articles by URL and title similarity.

    Parameters
    ----------
    threshold:
        Jaccard similarity above which two titles are considered duplicates.
    time_window_hours:
        Only compare articles published within this window of each other.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        time_window_hours: int = 6,
    ) -> None:
        self._threshold = threshold
        self._time_window = timedelta(hours=time_window_hours)

    # ── Single article check ─────────────────────────────────────────────────

    def is_duplicate(
        self,
        article: ParsedArticle,
        recent_articles: list[ParsedArticle],
        threshold: float | None = None,
    ) -> bool:
        """
        Check *article* against *recent_articles* for duplicates.

        Returns True if a duplicate is found.
        """
        threshold = threshold if threshold is not None else self._threshold
        article_words = _title_words(article.title)

        for existing in recent_articles:
            # Exact URL match
            if article.url == existing.url:
                return True

            # Title similarity within time window
            time_diff = abs(article.published_at - existing.published_at)
            if time_diff > self._time_window:
                continue

            similarity = _jaccard(article_words, _title_words(existing.title))
            if similarity > threshold:
                return True

        return False

    # ── Batch deduplication ──────────────────────────────────────────────────

    def deduplicate_batch(
        self, articles: list[ParsedArticle]
    ) -> list[ParsedArticle]:
        """
        Deduplicate a list of articles, keeping the most credible version.

        Articles are sorted by ``source_credibility`` descending before
        processing, so the first (most credible) version of each story wins.
        """
        sorted_articles = sorted(
            articles, key=lambda a: a.source_credibility, reverse=True
        )
        kept: list[ParsedArticle] = []

        for article in sorted_articles:
            if not self.is_duplicate(article, kept):
                kept.append(article)

        logger.info(
            "Deduplicated: %d articles -> %d unique", len(articles), len(kept)
        )
        return kept

    # ── DB-level deduplication ───────────────────────────────────────────────

    def is_duplicate_in_db(
        self,
        article: ParsedArticle,
        db: "DatabaseManager",
    ) -> bool:
        """
        Check if *article* already exists in the ``news_articles`` table.

        First checks exact URL match, then falls back to title similarity
        against the most recent 100 articles.
        """
        # Exact URL check
        row = db.fetch_one(
            "SELECT id FROM news_articles WHERE url = ?",
            (article.url,),
        )
        if row:
            return True

        # Title similarity against recent articles
        recent_rows = db.fetch_all(
            "SELECT title, published_at FROM news_articles "
            "ORDER BY published_at DESC LIMIT 100",
        )
        article_words = _title_words(article.title)
        for row in recent_rows:
            similarity = _jaccard(article_words, _title_words(row["title"]))
            if similarity > self._threshold:
                return True

        return False
