"""
WebSocket management for real-time task progress notifications.
"""

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

    async def connect(self, websocket: WebSocket, tenant_id: str) -> None:
        """WebSocket 연결 수락."""
        await websocket.accept()

        if tenant_id not in self._connections:
            self._connections[tenant_id] = []
        self._connections[tenant_id].append(websocket)
        self._subscriptions[websocket] = set()

        logger.info(f"WebSocket connected: tenant={tenant_id}")

        # 연결 확인 메시지 전송
        await websocket.send_json({"type": "connected", "tenant_id": tenant_id})

    def disconnect(self, websocket: WebSocket, tenant_id: str) -> None:
        """WebSocket 연결 해제."""
        if tenant_id in self._connections:
            if websocket in self._connections[tenant_id]:
                self._connections[tenant_id].remove(websocket)

            if not self._connections[tenant_id]:
                del self._connections[tenant_id]

        if websocket in self._subscriptions:
            del self._subscriptions[websocket]

        logger.info(f"WebSocket disconnected: tenant={tenant_id}")

    def subscribe(self, websocket: WebSocket, task_id: str) -> None:
        """특정 Task 구독."""
        if websocket in self._subscriptions:
            self._subscriptions[websocket].add(task_id)
            logger.debug(f"Subscribed to task: {task_id}")

    def unsubscribe(self, websocket: WebSocket, task_id: str) -> None:
        """특정 Task 구독 해제."""
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
        if tenant_id not in self._connections:
            return

        message_text = json.dumps(message)
        connections = self._connections[tenant_id].copy()
        disconnected = []

        for websocket in connections:
            # 해당 task를 구독 중이거나 모든 task를 구독 중인 경우 전송
            subscriptions = self._subscriptions.get(websocket, set())
            if task_id in subscriptions or "*" in subscriptions:
                try:
                    await websocket.send_text(message_text)
                except Exception as e:
                    logger.warning(f"WebSocket send error: {e}")
                    disconnected.append(websocket)

        # 끊어진 연결 정리
        for ws in disconnected:
            self.disconnect(ws, tenant_id)

    async def broadcast_to_tenant(
        self, tenant_id: str, message: Dict[str, Any]
    ) -> None:
        """테넌트의 모든 연결에 브로드캐스트."""
        if tenant_id not in self._connections:
            return

        message_text = json.dumps(message)
        connections = self._connections[tenant_id].copy()
        disconnected = []

        for websocket in connections:
            try:
                await websocket.send_text(message_text)
            except Exception:
                disconnected.append(websocket)

        for ws in disconnected:
            self.disconnect(ws, tenant_id)

    def get_connection_count(self, tenant_id: str = None) -> int:
        """연결 수 조회."""
        if tenant_id:
            return len(self._connections.get(tenant_id, []))
        return sum(len(conns) for conns in self._connections.values())


# 전역 인스턴스
task_ws_manager = TaskWebSocketManager()

# Backward-compatibility alias: routes.py previously imported this as `WebSocketManager`.
# The alias is kept here so existing code that may still reference it at runtime
# continues to work without breaking changes. routes.py has been updated to import
# TaskWebSocketManager directly; this alias can be removed once all callers are updated.
WebSocketManager = TaskWebSocketManager
