"""
Data models for Orange3 Web Backend
Based on orange-canvas-core scheme models
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import IntEnum, Enum
import uuid


# ============================================================================
# Enums
# ============================================================================

class NodeState(IntEnum):
    """Node runtime state flags (from SchemeNode.State)"""
    NoState = 0
    Running = 1
    Pending = 2
    Invalidated = 4
    NotReady = 8


class LinkState(str, Enum):
    """Link state"""
    Active = "Active"
    Pending = "Pending"


class MessageSeverity(IntEnum):
    """User message severity (from UserMessage)"""
    Info = 1
    Warning = 2
    Error = 3


class ContentType(str, Enum):
    """Annotation content type"""
    TextPlain = "text/plain"
    TextHtml = "text/html"
    TextRst = "text/rst"


# ============================================================================
# Position & Geometry
# ============================================================================

class Position(BaseModel):
    """Position of an element on the canvas."""
    x: float
    y: float


class Rect(BaseModel):
    """Rectangle geometry."""
    x: float
    y: float
    width: float
    height: float


# ============================================================================
# Channels (Inputs/Outputs)
# ============================================================================

class InputChannel(BaseModel):
    """Input channel definition (from InputSignal)."""
    id: str
    name: str
    types: List[str] = []  # Fully qualified type names
    flags: int = 0  # Signal flags
    multiple: bool = False  # Can accept multiple connections


class OutputChannel(BaseModel):
    """Output channel definition (from OutputSignal)."""
    id: str
    name: str
    types: List[str] = []  # Fully qualified type names
    flags: int = 0  # Signal flags
    dynamic: bool = False  # Dynamic type output


# ============================================================================
# Widget Registry
# ============================================================================

class WidgetDescription(BaseModel):
    """Widget type description (from WidgetDescription)."""
    id: str
    name: str
    description: str = ""
    long_description: str = ""
    icon: str = ""
    category: str = "Data"
    keywords: List[str] = []
    inputs: List[InputChannel] = []
    outputs: List[OutputChannel] = []
    priority: int = 0
    background: str = ""  # Category color


class WidgetCategory(BaseModel):
    """Widget category."""
    name: str
    description: str = ""
    color: str = "#808080"
    priority: int = 0
    widgets: List[WidgetDescription] = []


# ============================================================================
# Node
# ============================================================================

class UserMessage(BaseModel):
    """A user message displayed on a node (from UserMessage)."""
    content: str
    severity: MessageSeverity = MessageSeverity.Info
    message_id: str = ""
    data: Dict[str, Any] = {}


class WorkflowNode(BaseModel):
    """A node in the workflow (from SchemeNode)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    widget_id: str  # Reference to WidgetDescription.id
    title: str
    position: Position
    properties: Dict[str, Any] = {}  # Widget settings
    state: NodeState = NodeState.NoState
    progress: float = -1.0  # -1 = no progress, 0-100 = progress
    messages: List[UserMessage] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class NodeCreate(BaseModel):
    """Schema for creating a node."""
    widget_id: str
    title: str
    position: Position
    properties: Dict[str, Any] = {}


class NodeUpdate(BaseModel):
    """Schema for updating a node."""
    title: Optional[str] = None
    position: Optional[Position] = None
    properties: Optional[Dict[str, Any]] = None
    state: Optional[NodeState] = None
    progress: Optional[float] = None


# ============================================================================
# Link
# ============================================================================

class WorkflowLink(BaseModel):
    """A link between two nodes (from SchemeLink)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str
    source_channel: str  # Output channel name
    sink_node_id: str
    sink_channel: str  # Input channel name
    enabled: bool = True
    state: LinkState = LinkState.Active
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LinkCreate(BaseModel):
    """Schema for creating a link."""
    source_node_id: str
    source_channel: str
    sink_node_id: str
    sink_channel: str


class LinkUpdate(BaseModel):
    """Schema for updating a link."""
    enabled: Optional[bool] = None


class LinkValidation(BaseModel):
    """Schema for validating link compatibility."""
    source_node_id: str
    source_channel: str
    sink_node_id: str
    sink_channel: str


class LinkCompatibility(BaseModel):
    """Result of link compatibility check."""
    compatible: bool
    strict: bool = False  # Strict type match
    dynamic: bool = False  # Dynamic type match
    reason: Optional[str] = None


# ============================================================================
# Annotations
# ============================================================================

class TextAnnotation(BaseModel):
    """Text annotation (from SchemeTextAnnotation)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "text"
    rect: Rect
    content: str = ""
    content_type: ContentType = ContentType.TextPlain
    font: Dict[str, Any] = {}  # family, size, etc.


class ArrowAnnotation(BaseModel):
    """Arrow annotation (from SchemeArrowAnnotation)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "arrow"
    start_pos: Position
    end_pos: Position
    color: str = "#808080"


Annotation = Union[TextAnnotation, ArrowAnnotation]


class AnnotationCreate(BaseModel):
    """Schema for creating an annotation."""
    type: str  # "text" or "arrow"
    # For text
    rect: Optional[Rect] = None
    content: Optional[str] = None
    content_type: Optional[ContentType] = None
    font: Optional[Dict[str, Any]] = None
    # For arrow
    start_pos: Optional[Position] = None
    end_pos: Optional[Position] = None
    color: Optional[str] = None


class AnnotationUpdate(BaseModel):
    """Schema for updating an annotation."""
    # For text
    rect: Optional[Rect] = None
    content: Optional[str] = None
    content_type: Optional[ContentType] = None
    font: Optional[Dict[str, Any]] = None
    # For arrow
    start_pos: Optional[Position] = None
    end_pos: Optional[Position] = None
    color: Optional[str] = None


# ============================================================================
# Workflow
# ============================================================================

class WorkflowSummary(BaseModel):
    """Summary of a workflow for listing."""
    id: str
    tenant_id: str
    title: str
    description: str = ""
    node_count: int = 0
    link_count: int = 0
    created_at: datetime
    updated_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Workflow(BaseModel):
    """A complete workflow (from Scheme)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    title: str = "Untitled"
    description: str = ""
    env: Dict[str, Any] = {}  # Runtime environment
    nodes: List[WorkflowNode] = []
    links: List[WorkflowLink] = []
    annotations: List[Union[TextAnnotation, ArrowAnnotation]] = []
    loop_flags: int = 0  # NoLoops, AllowLoops, AllowSelfLoops
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_summary(self) -> WorkflowSummary:
        return WorkflowSummary(
            id=self.id,
            tenant_id=self.tenant_id,
            title=self.title,
            description=self.description,
            node_count=len(self.nodes),
            link_count=len(self.links),
            created_at=self.created_at,
            updated_at=self.updated_at
        )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WorkflowCreate(BaseModel):
    """Schema for creating a workflow."""
    title: str = "Untitled"
    description: str = ""


class WorkflowUpdate(BaseModel):
    """Schema for updating a workflow."""
    title: Optional[str] = None
    description: Optional[str] = None
    env: Optional[Dict[str, Any]] = None


# ============================================================================
# Execution
# ============================================================================

class ExecutionMode(str, Enum):
    """Workflow execution mode."""
    Full = "full"  # Execute entire workflow
    Partial = "partial"  # Execute from a specific node
    Node = "node"  # Execute single node


class ExecutionRequest(BaseModel):
    """Request to execute a workflow."""
    mode: ExecutionMode = ExecutionMode.Full
    target_node_id: Optional[str] = None  # For partial/node execution


class ExecutionStatus(BaseModel):
    """Workflow execution status."""
    workflow_id: str
    running: bool
    current_node_id: Optional[str] = None
    progress: float = 0.0
    errors: List[str] = []


# ============================================================================
# Tenant (Multi-tenancy)
# ============================================================================

class Tenant(BaseModel):
    """A tenant in the multi-tenant system."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# WebSocket Events
# ============================================================================

class WSEventType(str, Enum):
    """WebSocket event types."""
    # Server -> Client
    NodeStateChanged = "node_state_changed"
    NodeProgressChanged = "node_progress_changed"
    NodeMessageAdded = "node_message_added"
    LinkStateChanged = "link_state_changed"
    WorkflowChanged = "workflow_changed"
    SignalPropagated = "signal_propagated"
    ExecutionStarted = "execution_started"
    ExecutionFinished = "execution_finished"
    ExecutionError = "execution_error"
    # Client -> Server
    RunWorkflow = "run_workflow"
    StopWorkflow = "stop_workflow"
    ActivateNode = "activate_node"
    CursorMove = "cursor_move"
    NodeMove = "node_move"


class WSEvent(BaseModel):
    """WebSocket event."""
    type: WSEventType
    data: Dict[str, Any] = {}
