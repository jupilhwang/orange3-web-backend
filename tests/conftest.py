"""
Pytest fixtures for Orange3 Web Backend Tests
"""
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from httpx import AsyncClient, ASGITransport
import os
import sys
import tempfile

# 테스트용 SQLite DB 설정 (import 전에 환경변수 설정)
_test_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_test_db_file.name}"
os.environ["TASK_WORKER_ENABLED"] = "false"  # 테스트 시 워커 비활성화

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.core.database import init_db, close_db


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def setup_test_db():
    """테스트 DB 초기화."""
    await init_db()
    yield
    await close_db()
    # 테스트 DB 파일 삭제
    try:
        os.unlink(_test_db_file.name)
    except Exception:
        pass


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client for testing API endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_iris_data() -> dict:
    """Sample Iris dataset metadata for testing."""
    return {
        "name": "iris",
        "instances": 150,
        "features": 4,
        "target": "iris",
        "columns": [
            {"name": "sepal length", "type": "numeric"},
            {"name": "sepal width", "type": "numeric"},
            {"name": "petal length", "type": "numeric"},
            {"name": "petal width", "type": "numeric"},
            {"name": "iris", "type": "categorical"}
        ]
    }


@pytest.fixture
def sample_workflow() -> dict:
    """Sample workflow data for testing."""
    return {
        "id": "test-workflow-1",
        "name": "Test Workflow",
        "nodes": [
            {
                "id": "node-1",
                "widget_id": "file",
                "position": {"x": 100, "y": 100}
            },
            {
                "id": "node-2",
                "widget_id": "data-table",
                "position": {"x": 300, "y": 100}
            }
        ],
        "links": [
            {
                "id": "link-1",
                "source": "node-1",
                "target": "node-2",
                "source_port": "data",
                "target_port": "data"
            }
        ]
    }


@pytest.fixture
def sample_sample_request() -> dict:
    """Sample data sampling request for testing."""
    return {
        "data_path": "iris",
        "sampling_type": "fixed_proportion",
        "proportion": 0.7,
        "sample_size": None,
        "n_folds": 10,
        "selected_fold": 1,
        "use_seed": True,
        "seed": 42,
        "stratify": True,
        "with_replacement": False
    }

