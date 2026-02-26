"""
WebSocket management for real-time task progress notifications.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from fastapi import WebSocket
import json
import logging

logger = logging.getLogger(__name__)


class TaskWebSocketManager:
    """
    Task 진행률 및 완료 알림을 위한 WebSocket 매니저.

    - tenant_id별로 연결 관리
    - task_id를 구독하여 해당 작업의 알림만 수신
    """

    def __init__(self) -> None:
        # tenant_id -> list of WebSocket connections
        self._connections: Dict[str, List[WebSocket]] = {}
        # websocket -> set of subscribed task_ids
        self._subscriptions: Dict[WebSocket, Set[str]] = {}
        # Protects concurrent access to _connections and _subscriptions
        self._lock: asyncio.Lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, tenant_id: str) -> None:
        """WebSocket 연결 수락."""
        await websocket.accept()

        async with self._lock:
            if tenant_id not in self._connections:
                self._connections[tenant_id] = []
            self._connections[tenant_id].append(websocket)
            self._subscriptions[websocket] = set()

        logger.info(f"WebSocket connected: tenant={tenant_id}")

        # 연결 확인 메시지 전송
        await websocket.send_json({"type": "connected", "tenant_id": tenant_id})

    async def disconnect(self, websocket: WebSocket, tenant_id: str) -> None:
        """WebSocket 연결 해제."""
        async with self._lock:
            if tenant_id in self._connections:
                try:
                    self._connections[tenant_id].remove(websocket)
                except ValueError:
                    pass

                if not self._connections[tenant_id]:
                    del self._connections[tenant_id]

            self._subscriptions.pop(websocket, None)

        logger.info(f"WebSocket disconnected: tenant={tenant_id}")

    async def subscribe(self, websocket: WebSocket, task_id: str) -> None:
        """특정 Task 구독."""
        async with self._lock:
            if websocket in self._subscriptions:
                self._subscriptions[websocket].add(task_id)
        logger.debug(f"Subscribed to task: {task_id}")

    async def unsubscribe(self, websocket: WebSocket, task_id: str) -> None:
        """특정 Task 구독 해제."""
        async with self._lock:
            if websocket in self._subscriptions:
                self._subscriptions[websocket].discard(task_id)

    async def send_progress(
        self, task_id: str, tenant_id: str, progress: float, message: str = None
    ) -> None:
        """Task 진행률 알림 전송."""
        await self._broadcast_to_subscribers(
            task_id,
            tenant_id,
            {
                "type": "task_progress",
                "task_id": task_id,
                "progress": progress,
                "message": message,
            },
        )

    async def send_completion(
        self,
        task_id: str,
        tenant_id: str,
        status: str,
        result: Any = None,
        error: str = None,
    ) -> None:
        """Task 완료/실패 알림 전송."""
        await self._broadcast_to_subscribers(
            task_id,
            tenant_id,
            {
                "type": "task_completed",
                "task_id": task_id,
                "status": status,
                "result": result,
                "error": error,
            },
        )

    async def _broadcast_to_subscribers(
        self, task_id: str, tenant_id: str, message: Dict[str, Any]
    ) -> None:
        """구독자에게 메시지 브로드캐스트."""
        # 락을 최소 범위로 유지: 스냅샷 복사 후 락 해제, send는 락 외부에서 실행
        async with self._lock:
            if tenant_id not in self._connections:
                return
            connections_snapshot = list(self._connections[tenant_id])
            subscriptions_snapshot = {
                ws: set(subs) for ws, subs in self._subscriptions.items()
            }

        message_text = json.dumps(message)
        disconnected = []

        for websocket in connections_snapshot:
            subscriptions = subscriptions_snapshot.get(websocket, set())
            if task_id in subscriptions or "*" in subscriptions:
                try:
                    await websocket.send_text(message_text)
                except Exception as e:
                    logger.warning(f"WebSocket send error: {e}")
                    disconnected.append(websocket)

        for ws in disconnected:
            await self.disconnect(ws, tenant_id)

    async def broadcast_to_tenant(
        self, tenant_id: str, message: Dict[str, Any]
    ) -> None:
        """테넌트의 모든 연결에 브로드캐스트."""
        async with self._lock:
            if tenant_id not in self._connections:
                return
            connections_snapshot = list(self._connections[tenant_id])

        message_text = json.dumps(message)
        disconnected = []

        for websocket in connections_snapshot:
            try:
                await websocket.send_text(message_text)
            except Exception:
                disconnected.append(websocket)

        for ws in disconnected:
            await self.disconnect(ws, tenant_id)

    async def broadcast_to_workflow(
        self,
        workflow_id: str,
        message: Dict[str, Any],
        exclude: Optional[WebSocket] = None,
    ) -> None:
        """특정 workflow를 구독한 모든 연결에 브로드캐스트.

        연결은 tenant_id 기준으로 관리되므로, 전체 연결 중
        해당 workflow_id를 구독한 소켓에 전달합니다.

        Args:
            workflow_id: 브로드캐스트 대상 workflow ID
            message: 전송할 JSON 직렬화 가능 메시지
            exclude: 제외할 WebSocket 연결 (보통 메시지 발신자)
        """
        async with self._lock:
            targets: List[tuple] = []
            for tenant_id, ws_list in self._connections.items():
                for ws in ws_list:
                    if ws is exclude:
                        continue
                    subs = self._subscriptions.get(ws, set())
                    if workflow_id in subs or "*" in subs:
                        targets.append((tenant_id, ws))

        message_text = json.dumps(message)
        disconnected = []
        for tenant_id, ws in targets:
            try:
                await ws.send_text(message_text)
            except Exception as e:
                logger.warning(f"WebSocket send error in broadcast_to_workflow: {e}")
                disconnected.append((ws, tenant_id))

        for ws, tenant_id in disconnected:
            await self.disconnect(ws, tenant_id)

    async def get_connection_count(self, tenant_id: str) -> int:
        """테넌트의 WebSocket 연결 수 반환."""
        async with self._lock:
            return len(self._connections.get(tenant_id, []))


# 전역 인스턴스
task_ws_manager = TaskWebSocketManager()

# Backward-compatibility alias: routes.py previously imported this as `WebSocketManager`.
# The alias is kept here so existing code that may still reference it at runtime
# continues to work without breaking changes. routes.py has been updated to import
# TaskWebSocketManager directly; this alias can be removed once all callers are updated.
WebSocketManager = TaskWebSocketManager
