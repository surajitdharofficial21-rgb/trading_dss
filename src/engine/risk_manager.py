"""
Position sizing, stop-loss, and target calculation.

Based on configurable risk parameters and current market conditions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from config.constants import (
    STT_EQUITY_INTRADAY, EXCHANGE_TRANSACTION_CHARGE_NSE,
    SEBI_TURNOVER_FEE, GST_RATE, DEFAULT_BROKERAGE_FLAT,
    STT_FUTURES_SELL, EXCHANGE_TRANSACTION_CHARGE_NSE_FO,
    STAMP_DUTY_FO,
)

logger = logging.getLogger(__name__)


@dataclass
class PositionSizing:
    """
    Recommended position parameters for a trade.

    Attributes
    ----------
    quantity:
        Number of units/lots to trade.
    entry_price:
        Recommended entry price.
    stop_loss:
        Stop-loss price.
    target1:
        First profit target (1:1.5 R:R by default).
    target2:
        Second profit target (1:2.5 R:R).
    risk_per_unit:
        Rupee risk per unit (entry − stop_loss).
    total_risk:
        Total capital at risk (risk_per_unit × quantity × lot_size).
    rr_ratio:
        Risk-reward ratio for target1.
    """

    quantity: int
    entry_price: float
    stop_loss: float
    target1: float
    target2: float
    risk_per_unit: float
    total_risk: float
    rr_ratio: float


@dataclass
class TransactionCosts:
    """Breakdown of all-in transaction costs for a trade."""

    brokerage: float
    stt: float
    exchange_charges: float
    sebi_fee: float
    gst: float
    stamp_duty: float
    total: float
    total_pct: float  # total as % of trade value


class RiskManager:
    """
    Calculates position sizing and stop-loss levels.

    Parameters
    ----------
    capital:
        Total available trading capital (₹).
    risk_pct_per_trade:
        Maximum % of capital to risk per trade (default 1%).
    atr_sl_multiplier:
        Stop-loss = entry ± (ATR × this multiplier) (default 1.5).
    rr_target1:
        Risk-reward ratio for first target (default 1.5).
    rr_target2:
        Risk-reward ratio for second target (default 2.5).
    """

    def __init__(
        self,
        capital: float,
        risk_pct_per_trade: float = 1.0,
        atr_sl_multiplier: float = 1.5,
        rr_target1: float = 1.5,
        rr_target2: float = 2.5,
    ) -> None:
        self._capital = capital
        self._risk_pct = risk_pct_per_trade / 100
        self._atr_mult = atr_sl_multiplier
        self._rr1 = rr_target1
        self._rr2 = rr_target2

    def calculate_position(
        self,
        entry_price: float,
        atr: float,
        is_long: bool = True,
        lot_size: int = 1,
    ) -> PositionSizing:
        """
        Calculate recommended position size and levels.

        Parameters
        ----------
        entry_price:
            Expected entry price per unit.
        atr:
            Current Average True Range value for the instrument.
        is_long:
            ``True`` for buy/long, ``False`` for sell/short.
        lot_size:
            F&O lot size (use 1 for equity).

        Returns
        -------
        PositionSizing:
            Recommended position with all price levels.
        """
        sl_distance = self._atr_mult * atr
        max_risk_capital = self._capital * self._risk_pct
        risk_per_lot = sl_distance * lot_size
        quantity = max(1, int(max_risk_capital / risk_per_lot)) if risk_per_lot > 0 else 1

        if is_long:
            stop_loss = entry_price - sl_distance
            target1 = entry_price + sl_distance * self._rr1
            target2 = entry_price + sl_distance * self._rr2
        else:
            stop_loss = entry_price + sl_distance
            target1 = entry_price - sl_distance * self._rr1
            target2 = entry_price - sl_distance * self._rr2

        total_risk = risk_per_lot * quantity
        rr_ratio = self._rr1

        return PositionSizing(
            quantity=quantity,
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            target1=round(target1, 2),
            target2=round(target2, 2),
            risk_per_unit=round(sl_distance, 2),
            total_risk=round(total_risk, 2),
            rr_ratio=rr_ratio,
        )

    @staticmethod
    def calculate_fo_costs(
        trade_value: float,
        is_buy: bool = True,
    ) -> TransactionCosts:
        """
        Compute all-in transaction costs for an F&O trade.

        Parameters
        ----------
        trade_value:
            Notional value of the trade (price × lot_size × quantity).
        is_buy:
            ``True`` for buy leg, ``False`` for sell leg.

        Returns
        -------
        TransactionCosts:
            Itemised cost breakdown.
        """
        brokerage = DEFAULT_BROKERAGE_FLAT
        stt = trade_value * STT_FUTURES_SELL if not is_buy else 0.0
        exchange = trade_value * EXCHANGE_TRANSACTION_CHARGE_NSE_FO
        sebi = trade_value * SEBI_TURNOVER_FEE
        gst = (brokerage + exchange) * GST_RATE
        stamp = trade_value * STAMP_DUTY_FO if is_buy else 0.0
        total = brokerage + stt + exchange + sebi + gst + stamp
        total_pct = (total / trade_value * 100) if trade_value > 0 else 0.0

        return TransactionCosts(
            brokerage=round(brokerage, 2),
            stt=round(stt, 2),
            exchange_charges=round(exchange, 2),
            sebi_fee=round(sebi, 4),
            gst=round(gst, 2),
            stamp_duty=round(stamp, 2),
            total=round(total, 2),
            total_pct=round(total_pct, 4),
        )
