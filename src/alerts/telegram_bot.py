"""
Telegram alert delivery.

Sends formatted trading signals and anomaly notifications to a
configured Telegram chat/channel.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from config.settings import settings
from src.engine.decision_engine import TradingSignal, SignalDirection

logger = logging.getLogger(__name__)

_DIRECTION_EMOJI = {
    SignalDirection.BULLISH: "📈",
    SignalDirection.BEARISH: "📉",
    SignalDirection.NEUTRAL: "➡️",
}


class TelegramAlertError(Exception):
    """Raised when a Telegram notification fails irrecoverably."""


class TelegramBot:
    """
    Sends formatted alerts to Telegram.

    Requires ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` to be set
    in the environment. If not configured, all send methods log a warning
    and return silently.

    Parameters
    ----------
    bot_token:
        Telegram Bot API token. Defaults to ``settings.telegram.bot_token``.
    chat_id:
        Target chat or channel ID.  Defaults to ``settings.telegram.chat_id``.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        self._token = bot_token or settings.telegram.bot_token
        self._chat_id = chat_id or settings.telegram.chat_id
        self._bot = None  # lazy initialisation

    def _is_configured(self) -> bool:
        return bool(self._token and self._chat_id)

    def _get_bot(self):  # type: ignore[return]
        """Lazy-initialise the python-telegram-bot Application."""
        if not self._is_configured():
            return None
        if self._bot is None:
            try:
                from telegram import Bot
                self._bot = Bot(token=self._token)
            except ImportError:
                logger.error("python-telegram-bot not installed")
        return self._bot

    # ── Message formatting ────────────────────────────────────────────────────

    @staticmethod
    def format_signal(signal: TradingSignal) -> str:
        """
        Format a :class:`TradingSignal` as a Markdown alert message.

        Parameters
        ----------
        signal:
            Signal to format.

        Returns
        -------
        str:
            Telegram Markdown-formatted message string.
        """
        emoji = _DIRECTION_EMOJI.get(signal.direction, "➡️")
        confidence_bar = "█" * int(signal.confidence * 10) + "░" * (10 - int(signal.confidence * 10))
        reasons_text = "\n".join(f"  • {r}" for r in signal.reasons[:5])

        return (
            f"{emoji} *{signal.direction.value.upper()} SIGNAL* — `{signal.index_id}`\n\n"
            f"*Strength:* {signal.strength.value.replace('_', ' ').title()}\n"
            f"*Confidence:* {signal.confidence:.0%}  `{confidence_bar}`\n\n"
            f"*Scores:*\n"
            f"  Technical: `{signal.technical_score:+.3f}`\n"
            f"  Options:   `{signal.options_score:+.3f}`\n"
            f"  News:      `{signal.news_score:+.3f}`\n\n"
            f"*Reasons:*\n{reasons_text}\n\n"
            f"_VIX adjustment: {signal.vix_adjustment:.2f}_"
        )

    @staticmethod
    def format_anomaly(index_id: str, anomaly_type: str, description: str) -> str:
        """Format an anomaly alert message."""
        return (
            f"⚠️ *ANOMALY DETECTED* — `{index_id}`\n\n"
            f"*Type:* {anomaly_type.replace('_', ' ').title()}\n"
            f"*Detail:* {description}"
        )

    # ── Send methods ──────────────────────────────────────────────────────────

    async def send_text_async(self, message: str, parse_mode: str = "Markdown") -> bool:
        """
        Send a plain text message asynchronously.

        Parameters
        ----------
        message:
            Message text (supports Markdown if *parse_mode* is ``"Markdown"``).
        parse_mode:
            Telegram parse mode.

        Returns
        -------
        bool:
            ``True`` if sent successfully, ``False`` otherwise.
        """
        if not self._is_configured():
            logger.warning("Telegram not configured — skipping alert: %.80s…", message)
            return False

        bot = self._get_bot()
        if bot is None:
            return False

        try:
            await bot.send_message(
                chat_id=self._chat_id,
                text=message,
                parse_mode=parse_mode,
            )
            logger.debug("Telegram message sent (%d chars)", len(message))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to send Telegram message: %s", exc)
            return False

    def send_text(self, message: str) -> bool:
        """Synchronous wrapper around :meth:`send_text_async`."""
        return asyncio.run(self.send_text_async(message))

    async def send_signal_async(self, signal: TradingSignal) -> bool:
        """
        Format and send a trading signal alert.

        Only sends if the signal is actionable (confidence above threshold).

        Parameters
        ----------
        signal:
            Signal to deliver.
        """
        if not signal.is_actionable:
            logger.debug(
                "Signal for %s below alert threshold (%.2f) — not sending",
                signal.index_id, signal.confidence,
            )
            return False
        message = self.format_signal(signal)
        return await self.send_text_async(message)

    def send_signal(self, signal: TradingSignal) -> bool:
        """Synchronous wrapper around :meth:`send_signal_async`."""
        return asyncio.run(self.send_signal_async(signal))
