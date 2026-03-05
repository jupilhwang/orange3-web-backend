"""
Data loading endpoints — re-export shim.

All implementation has moved to:
- ``app.core.ssrf``        — SSRF validation & safe fetching
- ``app.api.data_router``  — endpoint handlers and helpers

This module re-exports the router (as both ``router`` and ``data_router``)
so that ``from app.api.data import data_router`` keeps working.
"""

from .data_router import router, UrlLoadRequest  # noqa: F401

# Backward-compatible alias used by app/main.py
data_router = router

__all__ = ["data_router", "router", "UrlLoadRequest"]
