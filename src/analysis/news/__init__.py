"""News ingestion sub-package: RSS fetching, parsing, deduplication, impact mapping, decay, and event calendar."""

from .rss_fetcher import RSSFetcher, RawArticle
from .article_parser import ArticleParser, ParsedArticle
from .deduplicator import ArticleDeduplicator
from .impact_mapper import (
    IndexImpact,
    MappedArticle,
    MarketNewsSummary,
    NewsImpactMapper,
)
from .time_decay import DecayedArticle, EffectiveNewsScore, TimeDecayEngine
from .event_calendar import EventCalendar, EventRegimeModifier, UpcomingEvent

__all__ = [
    "RSSFetcher",
    "RawArticle",
    "ArticleParser",
    "ParsedArticle",
    "ArticleDeduplicator",
    "IndexImpact",
    "MappedArticle",
    "MarketNewsSummary",
    "NewsImpactMapper",
    "DecayedArticle",
    "EffectiveNewsScore",
    "TimeDecayEngine",
    "EventCalendar",
    "EventRegimeModifier",
    "UpcomingEvent",
]
