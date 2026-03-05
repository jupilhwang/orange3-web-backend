"""
Health check and observability endpoints.

Provides /health, /health/ready, /health/live, /internal/metrics,
/internal/logs, /internal/resources, and WebSocket stats.
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..core import get_db
from ..core.config import get_config, get_upload_dir
from ..core.globals import (
    OTEL_AVAILABLE,
    SERVER_START_TIME,
    SERVER_VERSION,
    get_availability,
    get_telemetry,
)
from ..websocket_manager import task_ws_manager

logger = logging.getLogger(__name__)

health_router = APIRouter(tags=["Health"])


@health_router.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    availability = get_availability()
    config = get_config()
    storage_type = config.storage.type

    # Database storage types
    db_storage_types = {"sqlite", "mysql", "postgresql", "oracle", "database"}

    # Storage path depends on storage type
    if storage_type in db_storage_types:
        storage_path = config.database.url or "sqlite:///./orange3.db"
    else:
        storage_path = str(get_upload_dir())

    # Calculate uptime
    uptime_seconds = time.time() - SERVER_START_TIME

    # Determine database type from URL
    db_url = config.database.url or ""
    if "postgresql" in db_url or "postgres" in db_url:
        database_type = "postgresql"
    elif "mysql" in db_url:
        database_type = "mysql"
    elif "oracle" in db_url:
        database_type = "oracle"
    else:
        database_type = "sqlite"

    return {
        "status": "healthy",
        "service": "orange3-web-backend",
        "version": SERVER_VERSION,
        "uptime_seconds": round(uptime_seconds, 2),
        "orange3_available": availability.get("orange3", False),
        "database_type": database_type,
        "storage_type": storage_type,
        "storage_path": storage_path,
        "max_file_size_mb": config.storage.max_db_file_size // (1024 * 1024),
        "lock_type": "asyncio",
    }


@health_router.get("/internal/metrics")
async def get_metrics() -> dict:
    """Get OpenTelemetry metrics summary."""
    if OTEL_AVAILABLE and get_telemetry:
        telemetry = get_telemetry()
        if telemetry:
            return telemetry.get_metrics_summary()
    return {
        "service": "orange3-web-backend",
        "version": SERVER_VERSION,
        "otel_available": OTEL_AVAILABLE,
        "message": "OpenTelemetry not initialized",
    }


@health_router.get("/internal/logs")
async def get_logs(limit: int = 100) -> dict:
    """Get recent log entries."""
    if OTEL_AVAILABLE and get_telemetry:
        telemetry = get_telemetry()
        if telemetry:
            return telemetry.get_logs_response(limit)
    return {"service": "orange3-web-backend", "total": 0, "logs": []}


@health_router.get("/internal/resources")
async def get_resources() -> dict:
    """Get current resource usage (CPU, Memory, Disk, Throughput)."""
    if OTEL_AVAILABLE and get_telemetry:
        telemetry = get_telemetry()
        if telemetry:
            return telemetry.get_resource_usage()

    # Fallback without OpenTelemetry
    try:
        import psutil

        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return {
            "psutil_available": True,
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": memory.percent,
            "memory_used_mb": int(memory.used / (1024 * 1024)),
            "disk_percent": disk.percent,
            "disk_used_gb": round(disk.used / (1024 * 1024 * 1024), 2),
        }
    except ImportError:
        return {
            "psutil_available": False,
            "message": "psutil not installed. Run: pip install psutil",
        }


@health_router.get("/health/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)) -> dict:
    """Readiness probe - checks database connection."""
    from sqlalchemy import text

    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database not ready: {e}")


@health_router.get("/health/live")
async def liveness_check() -> dict:
    """Liveness probe - checks if service is alive."""
    return {"status": "alive"}


@health_router.get("/internal/ws-stats")
async def get_websocket_stats() -> dict:
    """WebSocket connection statistics."""
    return {"total_connections": task_ws_manager.get_connection_count(), "status": "ok"}
