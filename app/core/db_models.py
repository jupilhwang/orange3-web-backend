"""
SQLAlchemy ORM Models for Orange3 Web Backend
These models are used for database persistence with SQLite
"""

from datetime import datetime, timezone
from typing import Optional, List
import uuid
import json

from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    Index,
    LargeBinary,
)
from sqlalchemy.orm import relationship, Mapped, mapped_column, selectinload

from .database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class UserRole:
    """User role constants."""

    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class UserDB(Base):
    """
    User database model for authentication.
    Supports local email/password auth and OAuth (Google, GitHub).
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    password_hash: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # nullable for OAuth users
    name: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    # Role-based access control
    role: Mapped[str] = mapped_column(String(50), nullable=False, default=UserRole.USER)

    # OAuth provider fields
    google_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, index=True
    )
    github_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, index=True
    )

    # Account status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Refresh tokens: one-to-many relationship
    refresh_tokens: Mapped[List["RefreshTokenDB"]] = relationship(
        "RefreshTokenDB", back_populates="user", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_user_email", "email"),
        Index("idx_user_google_id", "google_id"),
        Index("idx_user_github_id", "github_id"),
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"


class RefreshTokenDB(Base):
    """
    Stored refresh tokens for invalidation support on logout.
    Expired tokens should be periodically cleaned up.
    """

    __tablename__ = "refresh_tokens"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )  # SHA256 hash
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    user: Mapped["UserDB"] = relationship("UserDB", back_populates="refresh_tokens")

    __table_args__ = (
        Index("idx_refresh_token_hash", "token_hash"),
        Index("idx_refresh_token_user", "user_id"),
        # Composite index for the common lookup: token_hash WHERE revoked=FALSE
        # Used by /refresh and /logout to validate tokens in one index scan
        Index("idx_refresh_token_hash_revoked", "token_hash", "revoked"),
        # Covering index for cleanup queries: find all active tokens per user
        Index("idx_refresh_token_user_revoked", "user_id", "revoked"),
    )

    def __repr__(self) -> str:
        return f"<RefreshToken(id={self.id}, user_id={self.user_id}, revoked={self.revoked})>"


class TaskStatus:
    """Task status constants."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority:
    """Task priority constants."""

    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


class TaskQueueDB(Base):
    """
    DB 독립적인 Task Queue 모델.
    PostgreSQL, MySQL, SQLite, Oracle 모두 지원.
    """

    __tablename__ = "task_queue"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    # Task 정보
    task_name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    task_args: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    task_kwargs: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON

    # 상태
    status: Mapped[str] = mapped_column(
        String(20), default=TaskStatus.PENDING, nullable=False, index=True
    )
    priority: Mapped[int] = mapped_column(
        Integer, default=TaskPriority.NORMAL, nullable=False
    )

    # 진행률 (0.0 ~ 100.0, -1은 미정)
    progress: Mapped[float] = mapped_column(Float, default=-1.0, nullable=False)
    progress_message: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # 결과
    result: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 재시도
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_retries: Mapped[int] = mapped_column(Integer, default=3, nullable=False)

    # 시간
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # 워커 정보
    worker_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # 인덱스
    __table_args__ = (
        Index("idx_task_status_priority", "status", "priority"),
        Index("idx_task_tenant_status", "tenant_id", "status"),
        # Full worker-claim index: status + priority DESC + created_at ASC
        # Supports: WHERE status='pending' ORDER BY priority DESC, created_at ASC
        Index("idx_task_claim", "status", "priority", "created_at"),
    )

    def __repr__(self):
        return f"<Task(id={self.id}, name={self.task_name}, status={self.status})>"


class FileStorageDB(Base):
    """
    File storage database model.
    Stores uploaded files in database (BLOB) for multi-server environments.
    Supports zlib compression for reduced storage size.
    """

    __tablename__ = "file_storage"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    # File metadata
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(
        String(100), default="application/octet-stream"
    )
    file_size: Mapped[int] = mapped_column(
        Integer, nullable=False
    )  # Stored size (compressed if applicable)
    original_size: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )  # Original size before compression
    checksum: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )  # SHA256 of original content

    # Compression flag (for backward compatibility with existing data)
    is_compressed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # File category: 'upload', 'corpus', 'dataset', etc.
    category: Mapped[str] = mapped_column(String(50), default="upload", index=True)

    # Actual file data (BLOB) - may be compressed
    file_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Indexes
    __table_args__ = (
        Index("idx_file_tenant_category", "tenant_id", "category"),
        Index("idx_file_filename", "filename"),
        # Composite index for the overwrite-check DELETE and file listing queries
        # Matches: WHERE tenant_id=? AND filename=? AND category=?
        Index("idx_file_tenant_filename_category", "tenant_id", "filename", "category"),
    )

    def __repr__(self):
        return f"<FileStorage(id={self.id}, filename={self.filename}, size={self.file_size})>"


class TenantDB(Base):
    """Tenant database model"""

    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    workflows: Mapped[List["WorkflowDB"]] = relationship(
        "WorkflowDB", back_populates="tenant", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Tenant(id={self.id}, name={self.name})>"


class WorkflowDB(Base):
    """Workflow database model"""

    __tablename__ = "workflows"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), default="Untitled")
    description: Mapped[str] = mapped_column(Text, default="")
    env: Mapped[Optional[str]] = mapped_column(Text, default="{}")  # JSON string
    loop_flags: Mapped[int] = mapped_column(Integer, default=0)
    version: Mapped[int] = mapped_column(Integer, default=1)  # Optimistic locking
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    tenant: Mapped["TenantDB"] = relationship("TenantDB", back_populates="workflows")
    nodes: Mapped[List["NodeDB"]] = relationship(
        "NodeDB", back_populates="workflow", cascade="all, delete-orphan"
    )
    links: Mapped[List["LinkDB"]] = relationship(
        "LinkDB", back_populates="workflow", cascade="all, delete-orphan"
    )
    annotations: Mapped[List["AnnotationDB"]] = relationship(
        "AnnotationDB", back_populates="workflow", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("idx_workflow_tenant", "tenant_id"),
        # Composite index for listing workflows sorted by update time (most common query)
        Index("idx_workflow_tenant_updated", "tenant_id", "updated_at"),
    )

    def get_env(self) -> dict:
        return json.loads(self.env) if self.env else {}

    def set_env(self, value: dict):
        self.env = json.dumps(value)

    def __repr__(self):
        return f"<Workflow(id={self.id}, title={self.title})>"


class NodeDB(Base):
    """Node database model"""

    __tablename__ = "nodes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    workflow_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("workflows.id"), nullable=False
    )
    widget_id: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    position_x: Mapped[float] = mapped_column(Float, default=0.0)
    position_y: Mapped[float] = mapped_column(Float, default=0.0)
    properties: Mapped[Optional[str]] = mapped_column(Text, default="{}")  # JSON string
    state: Mapped[int] = mapped_column(Integer, default=0)  # NodeState enum
    progress: Mapped[float] = mapped_column(Float, default=-1.0)
    messages: Mapped[Optional[str]] = mapped_column(Text, default="[]")  # JSON array
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    workflow: Mapped["WorkflowDB"] = relationship("WorkflowDB", back_populates="nodes")

    # Indexes
    __table_args__ = (Index("idx_node_workflow", "workflow_id"),)

    def get_properties(self) -> dict:
        return json.loads(self.properties) if self.properties else {}

    def set_properties(self, value: dict):
        self.properties = json.dumps(value)

    def get_messages(self) -> list:
        return json.loads(self.messages) if self.messages else []

    def set_messages(self, value: list):
        self.messages = json.dumps(value)

    def __repr__(self):
        return f"<Node(id={self.id}, title={self.title})>"


class LinkDB(Base):
    """Link database model"""

    __tablename__ = "links"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    workflow_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("workflows.id"), nullable=False
    )
    source_node_id: Mapped[str] = mapped_column(String(36), nullable=False)
    source_channel: Mapped[str] = mapped_column(String(255), nullable=False)
    sink_node_id: Mapped[str] = mapped_column(String(36), nullable=False)
    sink_channel: Mapped[str] = mapped_column(String(255), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    state: Mapped[str] = mapped_column(String(50), default="Active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    workflow: Mapped["WorkflowDB"] = relationship("WorkflowDB", back_populates="links")

    # Indexes
    __table_args__ = (
        Index("idx_link_workflow", "workflow_id"),
        Index("idx_link_source", "source_node_id"),
        Index("idx_link_sink", "sink_node_id"),
        # Composite index for duplicate-link check: all four routing columns
        Index(
            "idx_link_route",
            "workflow_id",
            "source_node_id",
            "source_channel",
            "sink_node_id",
            "sink_channel",
        ),
    )

    def __repr__(self):
        return f"<Link(id={self.id}, {self.source_node_id}->{self.sink_node_id})>"


class AnnotationDB(Base):
    """Annotation database model"""

    __tablename__ = "annotations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    workflow_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("workflows.id"), nullable=False
    )
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # "text" or "arrow"

    # For text annotations
    rect_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rect_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rect_width: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rect_height: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content_type: Mapped[Optional[str]] = mapped_column(
        String(50), default="text/plain"
    )
    font: Mapped[Optional[str]] = mapped_column(Text, default="{}")  # JSON string

    # For arrow annotations
    start_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    start_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    end_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    end_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    color: Mapped[Optional[str]] = mapped_column(String(50), default="#808080")

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    workflow: Mapped["WorkflowDB"] = relationship(
        "WorkflowDB", back_populates="annotations"
    )

    # Indexes
    __table_args__ = (Index("idx_annotation_workflow", "workflow_id"),)

    def get_font(self) -> dict:
        return json.loads(self.font) if self.font else {}

    def set_font(self, value: dict):
        self.font = json.dumps(value)

    def __repr__(self):
        return f"<Annotation(id={self.id}, type={self.type})>"


# ============================================================================
# Conversion functions between Pydantic and SQLAlchemy models
# ============================================================================


def workflow_db_to_pydantic(db_workflow: WorkflowDB) -> dict:
    """Convert WorkflowDB to Pydantic-compatible dict.

    NOTE: Always load *db_workflow* with eager loading to avoid N+1 queries:

        result = await db.execute(
            select(WorkflowDB)
            .where(WorkflowDB.id == workflow_id)
            .options(
                selectinload(WorkflowDB.nodes),
                selectinload(WorkflowDB.links),
                selectinload(WorkflowDB.annotations),
            )
        )
        db_workflow = result.scalar_one_or_none()
    """
    from .models import Position, Rect

    nodes = []
    for node in db_workflow.nodes:
        nodes.append(
            {
                "id": node.id,
                "widget_id": node.widget_id,
                "title": node.title,
                "position": {"x": node.position_x, "y": node.position_y},
                "properties": node.get_properties(),
                "state": node.state,
                "progress": node.progress,
                "messages": node.get_messages(),
                "created_at": node.created_at.isoformat(),
                "updated_at": node.updated_at.isoformat(),
            }
        )

    links = []
    for link in db_workflow.links:
        links.append(
            {
                "id": link.id,
                "source_node_id": link.source_node_id,
                "source_channel": link.source_channel,
                "sink_node_id": link.sink_node_id,
                "sink_channel": link.sink_channel,
                "enabled": link.enabled,
                "state": link.state,
                "created_at": link.created_at.isoformat(),
            }
        )

    annotations = []
    for ann in db_workflow.annotations:
        if ann.type == "text":
            annotations.append(
                {
                    "id": ann.id,
                    "type": "text",
                    "rect": {
                        "x": ann.rect_x,
                        "y": ann.rect_y,
                        "width": ann.rect_width,
                        "height": ann.rect_height,
                    },
                    "content": ann.content,
                    "content_type": ann.content_type,
                    "font": ann.get_font(),
                }
            )
        else:
            annotations.append(
                {
                    "id": ann.id,
                    "type": "arrow",
                    "start_pos": {"x": ann.start_x, "y": ann.start_y},
                    "end_pos": {"x": ann.end_x, "y": ann.end_y},
                    "color": ann.color,
                }
            )

    return {
        "id": db_workflow.id,
        "tenant_id": db_workflow.tenant_id,
        "title": db_workflow.title,
        "description": db_workflow.description,
        "env": db_workflow.get_env(),
        "loop_flags": db_workflow.loop_flags,
        "version": db_workflow.version,
        "nodes": nodes,
        "links": links,
        "annotations": annotations,
        "created_at": db_workflow.created_at.isoformat(),
        "updated_at": db_workflow.updated_at.isoformat(),
    }


def select_workflow_with_relations(workflow_id: str):
    """Return a SELECT statement that eager-loads all workflow relationships.

    Use this helper whenever fetching a single WorkflowDB that will be
    serialised via workflow_db_to_pydantic — it prevents N+1 queries by
    issuing one SELECT per related table (nodes, links, annotations) instead
    of one SELECT per row.

    Example::

        stmt = select_workflow_with_relations(workflow_id)
        result = await db.execute(stmt)
        db_workflow = result.scalar_one_or_none()
        data = workflow_db_to_pydantic(db_workflow)
    """
    from sqlalchemy import select as sa_select

    return (
        sa_select(WorkflowDB)
        .where(WorkflowDB.id == workflow_id)
        .options(
            selectinload(WorkflowDB.nodes),
            selectinload(WorkflowDB.links),
            selectinload(WorkflowDB.annotations),
        )
    )


def select_tenant_workflows_with_relations(tenant_id: str):
    """Return a SELECT statement that eager-loads all workflow relationships
    for every workflow belonging to a tenant.

    Use when listing all workflows for a tenant and each workflow needs its
    nodes/links/annotations (e.g. for bulk export or summary counts without
    lazy-loading).

    Example::

        stmt = select_tenant_workflows_with_relations(tenant_id)
        result = await db.execute(stmt)
        workflows = result.scalars().all()
    """
    from sqlalchemy import select as sa_select

    return (
        sa_select(WorkflowDB)
        .where(WorkflowDB.tenant_id == tenant_id)
        .order_by(WorkflowDB.updated_at.desc())
        .options(
            selectinload(WorkflowDB.nodes),
            selectinload(WorkflowDB.links),
            selectinload(WorkflowDB.annotations),
        )
    )
