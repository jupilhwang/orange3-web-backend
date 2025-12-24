"""
WebSocket management for real-time collaboration
"""
from typing import Dict, List, Optional, Any
from fastapi import WebSocket
import json


class WebSocketManager:
    """Manages WebSocket connections for real-time collaboration."""
    
    def __init__(self):
        # workflow_id -> list of connected websockets
        self._connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, workflow_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        if workflow_id not in self._connections:
            self._connections[workflow_id] = []
        self._connections[workflow_id].append(websocket)
        
        # Notify others about new connection
        await self.broadcast_to_workflow(
            workflow_id,
            {
                "type": "user_joined",
                "data": {"count": len(self._connections[workflow_id])}
            },
            exclude=websocket
        )
    
    def disconnect(self, websocket: WebSocket, workflow_id: str):
        """Remove a WebSocket connection."""
        if workflow_id in self._connections:
            if websocket in self._connections[workflow_id]:
                self._connections[workflow_id].remove(websocket)
            
            # Clean up empty workflow entries
            if not self._connections[workflow_id]:
                del self._connections[workflow_id]
    
    async def broadcast_to_workflow(
        self,
        workflow_id: str,
        message: Dict[str, Any],
        exclude: Optional[WebSocket] = None
    ):
        """Broadcast a message to all connections for a workflow."""
        if workflow_id not in self._connections:
            return
        
        message_text = json.dumps(message)
        
        # Create a copy of the list to avoid modification during iteration
        connections = self._connections[workflow_id].copy()
        disconnected = []
        
        for websocket in connections:
            if websocket == exclude:
                continue
            
            try:
                await websocket.send_text(message_text)
            except Exception:
                # Connection is broken, mark for removal
                disconnected.append(websocket)
        
        # Remove broken connections
        for ws in disconnected:
            self.disconnect(ws, workflow_id)
    
    def get_connection_count(self, workflow_id: str) -> int:
        """Get the number of connections for a workflow."""
        return len(self._connections.get(workflow_id, []))

