"""
Widget API routers for Orange3 Web.
Each widget has its own module with Request models and endpoints.
"""

from fastapi import APIRouter

# Import widget routers - Data
from .scatter_plot import router as scatter_plot_router
from .distributions import router as distributions_router
from .bar_plot import router as bar_plot_router
from .heat_map import router as heat_map_router
from .select_columns import router as select_columns_router
from .select_rows import router as select_rows_router
from .file_upload import router as file_upload_router
from .data_sampler import router as data_sampler_router
from .datasets import router as datasets_router

# Import widget routers - Model
from .knn import router as knn_router
from .tree import router as tree_router
from .naive_bayes import router as naive_bayes_router
from .logistic_regression import router as logistic_regression_router
from .random_forest import router as random_forest_router

__all__ = [
    # Data widgets
    "scatter_plot_router",
    "distributions_router",
    "bar_plot_router",
    "heat_map_router",
    "select_columns_router",
    "select_rows_router",
    "file_upload_router",
    "data_sampler_router",
    "datasets_router",
    # Model widgets
    "knn_router",
    "tree_router",
    "naive_bayes_router",
    "logistic_regression_router",
    "random_forest_router",
]

