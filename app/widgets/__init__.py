"""
Widget API routers for Orange3 Web.
Each widget has its own module with Request models and endpoints.
"""

from fastapi import APIRouter

# Import widget routers - Data
from .scatter_plot import router as scatter_plot_router
from .distributions import router as distributions_router
from .bar_plot import router as bar_plot_router
from .box_plot import router as box_plot_router
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
from .linear_regression import router as linear_regression_router

# Import widget routers - Evaluate
from .predictions import router as predictions_router
from .test_and_score import router as test_and_score_router
from .confusion_matrix import router as confusion_matrix_router

# Import widget routers - Unsupervised
from .kmeans import router as kmeans_router

# Import widget routers - Text Mining
from .corpus import router as corpus_router
from .preprocess_text import router as preprocess_text_router
from .bag_of_words import router as bag_of_words_router
from .word_cloud import router as word_cloud_router

# Import utility routers
from .data_info import router as data_info_router
from .feature_statistics import router as feature_statistics_router

__all__ = [
    # Data widgets
    "scatter_plot_router",
    "distributions_router",
    "bar_plot_router",
    "box_plot_router",
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
    "linear_regression_router",
    # Evaluate widgets
    "predictions_router",
    "test_and_score_router",
    "confusion_matrix_router",
    # Unsupervised widgets
    "kmeans_router",
    # Text Mining widgets
    "corpus_router",
    "preprocess_text_router",
    "bag_of_words_router",
    "word_cloud_router",
    # Utility routers
    "data_info_router",
    "feature_statistics_router",
]
