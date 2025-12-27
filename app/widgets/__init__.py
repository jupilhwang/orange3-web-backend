"""
Widget API routers for Orange3 Web.
Each widget has its own module with Request models and endpoints.
"""

from fastapi import APIRouter

# Import widget routers
from .scatter_plot import router as scatter_plot_router
from .distributions import router as distributions_router
from .bar_plot import router as bar_plot_router
from .heat_map import router as heat_map_router
from .select_columns import router as select_columns_router
from .select_rows import router as select_rows_router
from .file_upload import router as file_upload_router
from .data_sampler import router as data_sampler_router
from .datasets import router as datasets_router

__all__ = [
    "scatter_plot_router",
    "distributions_router",
    "bar_plot_router",
    "heat_map_router",
    "select_columns_router",
    "select_rows_router",
    "file_upload_router",
    "data_sampler_router",
    "datasets_router",
]

