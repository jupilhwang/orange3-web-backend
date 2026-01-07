"""
Task Queue API 테스트

Task Queue REST API 엔드포인트를 테스트합니다.
"""
import pytest
import sys
import os
from pathlib import Path
from httpx import AsyncClient, ASGITransport

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.core.db_models import TaskStatus, TaskPriority


@pytest.fixture
async def client():
    """Async HTTP client for testing API endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestTaskAPI:
    """Task API 엔드포인트 테스트"""
    
    @pytest.mark.asyncio
    async def test_get_registered_tasks(self, client: AsyncClient):
        """등록된 태스크 목록 조회 API"""
        response = await client.get("/api/v1/tasks/registered")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "tasks" in data
        assert "priorities" in data
        assert "statuses" in data
        
        # 기본 태스크들이 등록되어 있어야 함
        assert "ml.train_kmeans" in data["tasks"]
        assert "ml.train_classifier" in data["tasks"]
        assert "util.health_check" in data["tasks"]
    
    @pytest.mark.asyncio
    async def test_get_queue_stats(self, client: AsyncClient):
        """큐 통계 조회 API"""
        response = await client.get("/api/v1/tasks/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "pending" in data
        assert "running" in data
        assert "completed" in data
        assert "failed" in data
        assert "total" in data
    
    @pytest.mark.asyncio
    async def test_create_health_check_task(self, client: AsyncClient):
        """헬스 체크 태스크 생성 API"""
        response = await client.post(
            "/api/v1/tasks/health-check",
            headers={"X-Tenant-ID": "test_tenant"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "task_id" in data
        assert data["status"] == TaskStatus.PENDING
        assert "message" in data
    
    @pytest.mark.asyncio
    async def test_create_generic_task(self, client: AsyncClient):
        """일반 태스크 생성 API"""
        response = await client.post(
            "/api/v1/tasks",
            json={
                "task_name": "util.health_check",
                "args": [],
                "kwargs": {},
                "priority": TaskPriority.NORMAL
            },
            headers={"X-Tenant-ID": "test_tenant"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "task_id" in data
        assert data["status"] == TaskStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_create_task_unknown_task_name(self, client: AsyncClient):
        """존재하지 않는 태스크 생성 시 에러"""
        response = await client.post(
            "/api/v1/tasks",
            json={
                "task_name": "nonexistent.task",
                "args": [],
                "kwargs": {}
            },
            headers={"X-Tenant-ID": "test_tenant"}
        )
        
        assert response.status_code == 400
        assert "Unknown task" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_get_task_status(self, client: AsyncClient):
        """태스크 상태 조회 API"""
        # 먼저 태스크 생성
        create_response = await client.post(
            "/api/v1/tasks/health-check",
            headers={"X-Tenant-ID": "test_tenant"}
        )
        task_id = create_response.json()["task_id"]
        
        # 상태 조회
        response = await client.get(
            f"/api/v1/tasks/{task_id}",
            headers={"X-Tenant-ID": "test_tenant"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == task_id
        assert data["task_name"] == "util.health_check"
        assert "status" in data
    
    @pytest.mark.asyncio
    async def test_get_task_status_not_found(self, client: AsyncClient):
        """존재하지 않는 태스크 조회"""
        response = await client.get(
            "/api/v1/tasks/nonexistent-task-id",
            headers={"X-Tenant-ID": "test_tenant"}
        )
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_list_tasks(self, client: AsyncClient):
        """태스크 목록 조회 API"""
        # 테스트용 태스크 생성
        await client.post(
            "/api/v1/tasks/health-check",
            headers={"X-Tenant-ID": "list_test_tenant"}
        )
        
        response = await client.get(
            "/api/v1/tasks",
            headers={"X-Tenant-ID": "list_test_tenant"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "tasks" in data
        assert "total" in data
        assert isinstance(data["tasks"], list)
    
    @pytest.mark.asyncio
    async def test_list_tasks_with_status_filter(self, client: AsyncClient):
        """상태 필터링된 태스크 목록 조회"""
        response = await client.get(
            "/api/v1/tasks?status=pending",
            headers={"X-Tenant-ID": "test_tenant"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # 모든 태스크가 pending 상태여야 함
        for task in data["tasks"]:
            assert task["status"] == TaskStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_cancel_task(self, client: AsyncClient):
        """태스크 취소 API"""
        # 먼저 태스크 생성
        create_response = await client.post(
            "/api/v1/tasks/health-check",
            headers={"X-Tenant-ID": "cancel_test_tenant"}
        )
        task_id = create_response.json()["task_id"]
        
        # 취소
        response = await client.post(
            f"/api/v1/tasks/{task_id}/cancel",
            headers={"X-Tenant-ID": "cancel_test_tenant"}
        )
        
        assert response.status_code == 200
        
        # 상태 확인
        status_response = await client.get(
            f"/api/v1/tasks/{task_id}",
            headers={"X-Tenant-ID": "cancel_test_tenant"}
        )
        assert status_response.json()["status"] == TaskStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_kmeans_convenience_api(self, client: AsyncClient):
        """K-Means 편의 API"""
        response = await client.post(
            "/api/v1/tasks/kmeans",
            json={
                "data_path": "datasets/iris.tab",
                "k": 3,
                "max_iterations": 100
            },
            headers={"X-Tenant-ID": "test_tenant"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "task_id" in data
        assert data["status"] == TaskStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_classifier_convenience_api(self, client: AsyncClient):
        """분류기 편의 API"""
        response = await client.post(
            "/api/v1/tasks/classifier",
            json={
                "data_path": "datasets/iris.tab",
                "model_type": "knn",
                "params": {"k": 5}
            },
            headers={"X-Tenant-ID": "test_tenant"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "task_id" in data
        assert data["status"] == TaskStatus.PENDING


class TestTaskAPIAdminEndpoints:
    """Task API 관리자 엔드포인트 테스트"""
    
    @pytest.mark.asyncio
    async def test_cleanup_stale_tasks(self, client: AsyncClient):
        """오래된 태스크 정리 API"""
        response = await client.post(
            "/api/v1/tasks/admin/cleanup-stale?timeout_minutes=30"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "reset_count" in data
    
    @pytest.mark.asyncio
    async def test_cleanup_old_tasks(self, client: AsyncClient):
        """오래된 완료 태스크 삭제 API"""
        response = await client.post(
            "/api/v1/tasks/admin/cleanup-old?days=7"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "deleted_count" in data


# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

