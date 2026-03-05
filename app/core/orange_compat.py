"""
Centralized Orange3 availability check.
Import ORANGE_AVAILABLE and Orange3 types from here instead of
having each widget file re-check independently.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

ORANGE_AVAILABLE = False
Table = None
Domain = None
DiscreteVariable = None
ContinuousVariable = None
StringVariable = None

try:
    from Orange.data import (
        Table,
        Domain,
        DiscreteVariable,
        ContinuousVariable,
        StringVariable,
    )

    ORANGE_AVAILABLE = True
    logger.debug("Orange3 is available")
except ImportError:
    logger.warning("Orange3 not available — running in mock mode")

__all__ = [
    "ORANGE_AVAILABLE",
    "Table",
    "Domain",
    "DiscreteVariable",
    "ContinuousVariable",
    "StringVariable",
]
