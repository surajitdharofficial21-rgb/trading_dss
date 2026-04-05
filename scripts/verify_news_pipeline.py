import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.news import RSSFetcher, ArticleParser, ArticleDeduplicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

def run_pipeline():
    try:
        # Fetch news
        print("--- Step 1: Fetching News ---")
        fetcher = RSSFetcher()
        # To avoid taking too long or hitting rate limits in a test, 
        # we can just fetch all feeds or a subset. 
        # fetch_all_feeds() respects refresh intervals, but since this is first run, 
        # it will fetch everything active.
        raw_articles = fetcher.fetch_all_feeds()
        print(f"Fetched: {len(raw_articles)} raw articles")
        stats = fetcher.get_fetch_stats()
        print(f"Stats: {stats}")

        if not raw_articles:
            print("No articles fetched. Check internet connection or feed URLs.")
            return

        # Parse
        print("\n--- Step 2: Parsing Articles ---")
        parser = ArticleParser()
        parsed = []
        for a in raw_articles:
            try:
                p = parser.parse_article(a)
                if p:
                    parsed.append(p)
            except Exception as e:
                print(f"Error parsing article {a.url}: {e}")
        
        print(f"Parsed: {len(parsed)} articles")

        # Show a sample
        print("\n--- Sample Articles (Max 3) ---")
        for i, a in enumerate(parsed[:3]):
            print(f"\n[{i+1}] Title: {a.title}")
            print(f"    Source: {a.source} (credibility: {a.source_credibility})")
            print(f"    Companies: {a.mentioned_companies}")
            print(f"    Sectors: {a.mentioned_sectors}")
            print(f"    Indices: {a.mentioned_indices}")
            print(f"    Event type: {a.event_type}")
            print(f"    Market Relevant: {a.is_market_hours_relevant}")

        # Deduplicate
        print("\n--- Step 3: Deduplication ---")
        dedup = ArticleDeduplicator()
        unique = dedup.deduplicate_batch(parsed)
        print(f"After dedup: {len(unique)} unique articles (removed {len(parsed) - len(unique)} duplicates)")

        # Validation logic
        print("\n--- Validation ---")
        if len(unique) > 0:
            print("✅ Pipeline executed successfully with data.")
        else:
            print("⚠️ Pipeline executed but no unique articles produced.")
            
        # Specific validation checks
        if any(a.mentioned_companies for a in unique):
            print("✅ Entity extraction (Companies) is working.")
        else:
            print("ℹ️ No companies identified in this batch (normal if no matches found).")
            
        if any(a.mentioned_sectors for a in unique):
            print("✅ Entity extraction (Sectors) is working.")
        else:
            print("ℹ️ No sectors identified in this batch.")

        if any(a.event_type for a in unique):
            print("✅ Event classification is working.")
        else:
            print("ℹ️ No events classified in this batch.")

    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_pipeline()
