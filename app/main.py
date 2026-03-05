"""
Orange3 Web Backend - Multi-tenant FastAPI Application

This backend REUSES the existing orange-canvas-core and orange-widget-base code.
It only adds a thin web API layer on top.

Features:
- SQLite database for persistence
- Async locks for concurrent access protection
- Multi-tenant support via X-Tenant-ID header
"""

import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Rate limiting — shared limiter for widget/auth routers
from .core.rate_limit import limiter

# Shared globals
from .core.globals import SERVER_VERSION

# Lifespan
from .core.lifespan import lifespan

# Middleware
from .middleware.proxy import setup_proxy_middleware

# API routers (extracted)
from .api.health import health_router
from .api.data import data_router

# Widget API routers
from .widgets import (
    scatter_plot_router,
    distributions_router,
    bar_plot_router,
    box_plot_router,
    heat_map_router,
    select_columns_router,
    select_rows_router,
    group_by_router,
    file_upload_router,
    data_sampler_router,
    datasets_router,
    knn_router,
    tree_router,
    naive_bayes_router,
    logistic_regression_router,
    random_forest_router,
    linear_regression_router,
    svm_router,
    neural_network_router,
    predictions_router,
    test_and_score_router,
    confusion_matrix_router,
    kmeans_router,
    pca_router,
    corpus_router,
    preprocess_text_router,
    bag_of_words_router,
    word_cloud_router,
    data_info_router,
    feature_statistics_router,
)
from .core.task_api import router as task_api_router
from .routes import (
    workflow_router,
    widget_registry_router,
    websocket_endpoint as workflow_ws_endpoint,
    legacy_widgets_handler,
)
from .websocket_manager import task_ws_manager

logger = logging.getLogger(__name__)


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Orange3 Web API",
    description="""
    Multi-tenant web-based Orange3 workflow canvas backend.
    
    **This backend REUSES existing orange-canvas-core and orange-widget-base code.**
    
    ## Features
    - SQLite database for persistence
    - Async locks for concurrent access protection
    - Multi-tenant support via X-Tenant-ID header
    
    ## Architecture
    - Wraps existing `Scheme`, `SchemeNode`, `SchemeLink` classes
    - Uses existing `WidgetRegistry` and `WidgetDescription`
    - Uses existing `compatible_channels` for type checking
    - Uses existing `scheme_to_ows_stream` for .ows export
    
    ## Authentication
    Use `X-Tenant-ID` header for multi-tenant access.
    """,
    version=SERVER_VERSION,
    lifespan=lifespan,
)

# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
_cors_env = os.environ.get("CORS_ALLOW_ORIGINS", "*")
cors_origins = [o.strip() for o in _cors_env.split(",")] if _cors_env != "*" else ["*"]

# Per CORS spec: wildcard origins cannot be used with credentials=True
_allow_credentials = "*" not in cors_origins
if not _allow_credentials:
    logger.warning(
        "CORS: allow_credentials disabled because allow_origins contains '*'. "
        "Set CORS_ALLOW_ORIGINS environment variable to specific origins "
        "if credentials are needed."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate-limiting middleware
app.add_middleware(SlowAPIMiddleware)

# Reverse proxy header support
setup_proxy_middleware(app)


# ============================================================================
# Health endpoints (registered directly on app — no /api/v1 prefix)
# ============================================================================

app.include_router(health_router)


# ============================================================================
# Task Progress WebSocket
# ============================================================================


@app.websocket("/ws/tasks/{tenant_id}")
async def task_websocket_endpoint(websocket: WebSocket, tenant_id: str) -> None:
    """Task progress and completion notification WebSocket."""
    await task_ws_manager.connect(websocket, tenant_id)

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            task_id = data.get("task_id")

            if action == "subscribe" and task_id:
                task_ws_manager.subscribe(websocket, task_id)
                await websocket.send_json({"type": "subscribed", "task_id": task_id})
            elif action == "unsubscribe" and task_id:
                task_ws_manager.unsubscribe(websocket, task_id)
                await websocket.send_json({"type": "unsubscribed", "task_id": task_id})
            elif action == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        task_ws_manager.disconnect(websocket, tenant_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        task_ws_manager.disconnect(websocket, tenant_id)


# ============================================================================
# Legacy Endpoints
# ============================================================================


@app.get("/api/widgets")
async def legacy_widgets() -> dict:
    """Legacy endpoint for frontend compatibility."""
    return await legacy_widgets_handler()


# ============================================================================
# API v1 Router
# ============================================================================

api_v1 = APIRouter(prefix="/api/v1")

# Data loading
api_v1.include_router(data_router)

# Widget routers
api_v1.include_router(scatter_plot_router)
api_v1.include_router(distributions_router)
api_v1.include_router(bar_plot_router)
api_v1.include_router(box_plot_router)
api_v1.include_router(heat_map_router)
api_v1.include_router(select_columns_router)
api_v1.include_router(select_rows_router)
api_v1.include_router(group_by_router)
api_v1.include_router(file_upload_router)
api_v1.include_router(data_sampler_router)
api_v1.include_router(datasets_router)
api_v1.include_router(knn_router)
api_v1.include_router(tree_router)
api_v1.include_router(naive_bayes_router)
api_v1.include_router(logistic_regression_router)
api_v1.include_router(random_forest_router)
api_v1.include_router(linear_regression_router)
api_v1.include_router(svm_router)
api_v1.include_router(neural_network_router)
api_v1.include_router(predictions_router)
api_v1.include_router(test_and_score_router)
api_v1.include_router(confusion_matrix_router)
api_v1.include_router(kmeans_router)
api_v1.include_router(pca_router)
api_v1.include_router(corpus_router)
api_v1.include_router(preprocess_text_router)
api_v1.include_router(bag_of_words_router)
api_v1.include_router(word_cloud_router)
api_v1.include_router(data_info_router)
api_v1.include_router(feature_statistics_router)

# Task queue
api_v1.include_router(task_api_router)

# Workflow
api_v1.include_router(workflow_router)

# Widget registry
api_v1.include_router(widget_registry_router)

# Auth
from .auth_routes import router as auth_router  # noqa: E402

api_v1.include_router(auth_router)

# Mount the versioned API
app.include_router(api_v1)


# WebSocket endpoint (registered directly on app, not api_v1)
@app.websocket("/api/v1/workflows/{workflow_id}/ws")
async def websocket_endpoint(websocket: WebSocket, workflow_id: str) -> None:
    """WebSocket for real-time updates."""
    await workflow_ws_endpoint(websocket, workflow_id)


if __name__ == "__main__":
    import uvicorn

    trusted_proxies = os.environ.get(
        "TRUSTED_PROXIES",
        "127.0.0.0/8,::1/128,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16",
    )
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        proxy_headers=True,
        forwarded_allow_ips=trusted_proxies,
    )
