"""
FastAPI application factory.

Create the app with ``create_app()`` or import the pre-built ``app``
singleton for ASGI deployment.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.logging_config import setup_logging
from config.settings import settings
from src.api.routes import indices, market_data, signals, backtest

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown logic."""
    setup_logging(
        log_dir=settings.logging.log_dir,
        console_level=getattr(logging, settings.logging.level),
        max_bytes=settings.logging.max_bytes,
        backup_count=settings.logging.backup_count,
    )
    logger.info("trading_dss API starting (env=%s)", settings.environment)
    yield
    logger.info("trading_dss API shutting down")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns
    -------
    FastAPI:
        Fully configured application instance.
    """
    application = FastAPI(
        title="trading_dss API",
        description="Trading Decision Support System for Indian Stock Markets",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    application.include_router(indices.router, prefix="/api/v1/indices", tags=["indices"])
    application.include_router(market_data.router, prefix="/api/v1/market", tags=["market"])
    application.include_router(signals.router, prefix="/api/v1/signals", tags=["signals"])
    application.include_router(backtest.router, prefix="/api/v1/backtest", tags=["backtest"])

    @application.get("/health", tags=["health"])
    async def health() -> dict:
        """Health check endpoint."""
        return {"status": "ok", "environment": settings.environment}

    return application


app = create_app()
