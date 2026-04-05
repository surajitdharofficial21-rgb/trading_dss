"""
Trading signal API endpoints.

GET /api/v1/signals/{index_id}   — generate a live signal for an index
GET /api/v1/signals/             — signals for all active F&O indices
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_index_registry
from src.data.index_registry import IndexRegistry

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/{index_id}", summary="Generate trading signal for an index")
async def get_signal(
    index_id: str,
    registry: IndexRegistry = Depends(get_index_registry),
) -> dict:
    """
    Generate a composite trading signal for *index_id*.

    This endpoint assembles available technical, options, and news data
    for the requested index and runs it through the decision engine.
    """
    defn = registry.get_or_none(index_id.upper())
    if defn is None:
        raise HTTPException(status_code=404, detail=f"Index not found: {index_id!r}")

    # Minimal response skeleton — full pipeline wiring done in run_data_collector
    return {
        "index_id": defn.id,
        "display_name": defn.display_name,
        "signal": "neutral",
        "confidence": 0.0,
        "message": "Signal engine requires data collection pipeline to be running",
    }


@router.get("/", summary="Signals for all active F&O indices")
async def list_signals(
    registry: IndexRegistry = Depends(get_index_registry),
) -> list[dict]:
    """Return signal stubs for all active F&O-enabled indices."""
    fo_indices = registry.filter(has_options=True, active_only=True)
    return [
        {
            "index_id": defn.id,
            "display_name": defn.display_name,
            "signal": "neutral",
            "confidence": 0.0,
        }
        for defn in fo_indices
    ]
