from src.analysis.news.time_decay import TimeDecayEngine
from src.analysis.news.event_calendar import EventCalendar
from datetime import datetime, timedelta
from src.utils.date_utils import get_ist_now

# Time decay test
decay = TimeDecayEngine()

now = get_ist_now()
# Fresh article
factor = decay.calculate_decay(now - timedelta(minutes=5), now, "CRITICAL", "POLICY")
print(f"5 min old CRITICAL POLICY: decay = {factor:.3f}")  # Should be ~0.97

# 2 hours old
factor = decay.calculate_decay(now - timedelta(hours=2), now, "HIGH", "EARNINGS")
print(f"2h old HIGH EARNINGS: decay = {factor:.3f}")  # Should be ~0.25

# Yesterday
factor = decay.calculate_decay(now - timedelta(hours=24), now, "CRITICAL", "POLICY")
print(f"24h old CRITICAL: decay = {factor:.3f}")  # Should be very low

# Event calendar
calendar = EventCalendar()
events = calendar.get_upcoming_events(days_ahead=7)
print(f"\nUpcoming events (next 7 days):")
for e in events:
    print(f"  [{e.impact_severity:>8}] {e.name} — {e.expected_date}")

modifier = calendar.get_regime_modifier("BANKNIFTY")
print(f"\nBANKNIFTY regime: {modifier.caution_level}")
print(f"Position modifier: {modifier.position_size_modifier}")
print(f"Reasoning: {modifier.reasoning}")
