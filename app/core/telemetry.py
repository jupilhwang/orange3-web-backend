"""
OpenTelemetry integration for Orange3-Web Backend.

Provides:
- Tracing: Distributed tracing with span creation
- Metrics: Request counts, latency, throughput, resource usage
- Logging: Structured logging with trace context
"""

import logging
import os
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request

# Optional psutil for resource monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
    ConsoleMetricExporter,
)
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.trace import Status, StatusCode

# Conditional OTLP imports
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False

from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogExporter
from opentelemetry._logs import set_logger_provider


# VERSION file path relative to this file: backend/app/core/telemetry.py → backend/VERSION
from app.core.config import _get_version


@dataclass
class TelemetryConfig:
    """Configuration for OpenTelemetry."""

    service_name: str = "orange3-web-backend"
    service_version: str = field(default_factory=_get_version)
    environment: str = "development"
    otel_endpoint: Optional[str] = None
    enable_console: bool = False
    log_level: str = "INFO"
    metric_interval_ms: int = 15000

    def get_otlp_endpoint(self) -> Optional[str]:
        """
        Returns the gRPC endpoint for OTLP exporters.

        Accepts either 'host:port' or 'http(s)://host:port' format.
        Returns plain 'host:port' for gRPC (no scheme needed).
        """
        if not self.otel_endpoint:
            return None
        endpoint = self.otel_endpoint.strip()
        # Strip scheme if present — gRPC endpoint is just host:port
        if endpoint.startswith("https://"):
            endpoint = endpoint[len("https://") :]
        elif endpoint.startswith("http://"):
            endpoint = endpoint[len("http://") :]
        return endpoint


@dataclass
class LogEntry:
    """Structured log entry."""

    timestamp: str
    level: str
    message: str
    service: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


class Telemetry:
    """Manages OpenTelemetry resources for the backend."""

    def __init__(self, config: TelemetryConfig):
        self.config = config
        self.tracer: Optional[trace.Tracer] = None
        self.meter: Optional[metrics.Meter] = None
        self._recent_logs: List[LogEntry] = []
        self._max_log_entries = 1000
        self._start_time = time.time()

        # Metrics
        self._request_counter = None
        self._request_duration = None
        self._active_requests = 0
        self._log_counter = None

        # Throughput tracking
        self._throughput_counter = None
        self._request_count_per_minute = 0
        self._last_minute_requests = 0
        self._throughput_lock = threading.Lock()

        # Backend metrics tracking
        self._metrics: Dict[str, Dict] = {}

        # Resource snapshots
        self._resource_snapshot = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "memory_used_mb": 0,
            "disk_percent": 0.0,
            "disk_used_gb": 0,
            "throughput_rpm": 0,  # Requests per minute
        }

    def setup(self, app: FastAPI) -> None:
        """Initialize OpenTelemetry with the FastAPI app."""
        # Create resource
        resource = Resource.create(
            {
                SERVICE_NAME: self.config.service_name,
                SERVICE_VERSION: self.config.service_version,
                "environment": self.config.environment,
            }
        )

        # Setup tracing
        self._setup_tracing(resource)

        # Setup metrics
        self._setup_metrics(resource)

        # Setup logging
        self._setup_logging(resource)

        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)

        logging.info(
            f"[OTel] Telemetry initialized for service: {self.config.service_name}"
        )

    def _setup_tracing(self, resource: Resource) -> None:
        """Configure trace provider."""
        provider = TracerProvider(resource=resource)

        endpoint = self.config.get_otlp_endpoint()
        if endpoint and OTLP_AVAILABLE:
            # OTLP gRPC exporter — endpoint is host:port, insecure for dev
            exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logging.info(f"[OTel] Tracing: OTLP gRPC exporter -> {endpoint}")
        elif self.config.enable_console:
            # Console exporter for debugging
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            logging.info("[OTel] Tracing: Console exporter")
        else:
            logging.info("[OTel] Tracing: No exporter (in-memory only)")

        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(self.config.service_name)

    def _setup_metrics(self, resource: Resource) -> None:
        """Configure meter provider."""
        readers = []

        endpoint = self.config.get_otlp_endpoint()
        if endpoint and OTLP_AVAILABLE:
            # OTLP gRPC exporter — endpoint is host:port, insecure for dev
            exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
            readers.append(
                PeriodicExportingMetricReader(
                    exporter, export_interval_millis=self.config.metric_interval_ms
                )
            )
            logging.info(f"[OTel] Metrics: OTLP gRPC exporter -> {endpoint}")
        elif self.config.enable_console:
            # Console exporter for debugging
            readers.append(
                PeriodicExportingMetricReader(
                    ConsoleMetricExporter(),
                    export_interval_millis=self.config.metric_interval_ms,
                )
            )
            logging.info("[OTel] Metrics: Console exporter")

        if readers:
            provider = MeterProvider(resource=resource, metric_readers=readers)
        else:
            provider = MeterProvider(resource=resource)
        metrics.set_meter_provider(provider)
        self.meter = metrics.get_meter(self.config.service_name)

        # Create metrics instruments
        self._request_counter = self.meter.create_counter(
            "http.requests.total", description="Total HTTP requests", unit="{requests}"
        )

        self._request_duration = self.meter.create_histogram(
            "http.request.duration",
            description="HTTP request duration in milliseconds",
            unit="ms",
        )

        self._log_counter = self.meter.create_counter(
            "logs.total", description="Total log entries", unit="{logs}"
        )

        # Throughput metric
        self._throughput_counter = self.meter.create_counter(
            "http.throughput.total",
            description="Total requests for throughput calculation",
            unit="{requests}",
        )

        # System resource metrics using ObservableGauge callbacks (psutil)
        if PSUTIL_AVAILABLE:

            def _cpu_callback(options):
                yield metrics.Observation(psutil.cpu_percent(interval=None))

            def _memory_callback(options):
                yield metrics.Observation(psutil.virtual_memory().percent)

            def _memory_used_mb_callback(options):
                yield metrics.Observation(psutil.virtual_memory().used / (1024 * 1024))

            def _disk_callback(options):
                yield metrics.Observation(psutil.disk_usage("/").percent)

            def _disk_used_gb_callback(options):
                yield metrics.Observation(
                    psutil.disk_usage("/").used / (1024 * 1024 * 1024)
                )

            self.meter.create_observable_gauge(
                "system.cpu.percent",
                callbacks=[_cpu_callback],
                description="CPU usage percentage",
                unit="%",
            )
            self.meter.create_observable_gauge(
                "system.memory.percent",
                callbacks=[_memory_callback],
                description="Memory usage percentage",
                unit="%",
            )
            self.meter.create_observable_gauge(
                "system.memory.used_mb",
                callbacks=[_memory_used_mb_callback],
                description="Memory used in megabytes",
                unit="MB",
            )
            self.meter.create_observable_gauge(
                "system.disk.percent",
                callbacks=[_disk_callback],
                description="Disk usage percentage",
                unit="%",
            )
            self.meter.create_observable_gauge(
                "system.disk.used_gb",
                callbacks=[_disk_used_gb_callback],
                description="Disk used in gigabytes",
                unit="GB",
            )
            logging.info(
                "[OTel] System resource metrics (CPU/Memory/Disk) registered via ObservableGauge"
            )
        else:
            logging.warning(
                "[OTel] psutil not available, system resource metrics disabled"
            )

    def _setup_logging(self, resource: Resource) -> None:
        """Configure logging with OTLP export."""
        # Create LoggerProvider
        provider = LoggerProvider(resource=resource)

        # Add OTLP Exporter if endpoint is configured
        endpoint = self.config.get_otlp_endpoint()
        if endpoint and OTLP_AVAILABLE:
            # OTLP gRPC exporter — endpoint is host:port, insecure for dev
            exporter = OTLPLogExporter(endpoint=endpoint, insecure=True)
            provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
            logging.info(f"[OTel] Logging: OTLP gRPC exporter -> {endpoint}")
        elif self.config.enable_console:
            provider.add_log_record_processor(
                BatchLogRecordProcessor(ConsoleLogExporter())
            )
            logging.info("[OTel] Logging: Console exporter")
        else:
            logging.info("[OTel] Logging: No exporter (in-memory only)")

        set_logger_provider(provider)

        # Instrument Python logging to include trace context
        try:
            LoggingInstrumentor().instrument(set_logging_format=True)
        except Exception as e:
            logging.warning(f"LoggingInstrumentor failed: {e}")

        # Setup custom log handler
        handler = OTelLogHandler(self)

        # Setup OTLP Log handler
        otlp_handler = LoggingHandler(level=logging.NOTSET, logger_provider=provider)

        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)

        # Formatters
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        otlp_handler.setFormatter(formatter)

        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(handler)
        root_logger.addHandler(otlp_handler)

        # Also add to uvicorn loggers
        for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"]:
            logger = logging.getLogger(logger_name)
            logger.addHandler(handler)
            logger.addHandler(otlp_handler)

        logging.info(f"[OTel] Logging initialized with level: {self.config.log_level}")

    def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Record HTTP request metrics."""
        status = (
            "success"
            if status_code < 400
            else ("client_error" if status_code < 500 else "error")
        )

        attributes = {
            "http.method": method,
            "http.route": path,
            "http.status_code": status_code,
            "status": status,
        }

        if self._request_counter:
            self._request_counter.add(1, attributes)
        if self._request_duration:
            self._request_duration.record(duration_ms, attributes)

        # Update throughput counter
        with self._throughput_lock:
            self._request_count_per_minute += 1

    def add_log_entry(self, entry: LogEntry) -> None:
        """Add a log entry to the buffer."""
        self._recent_logs.append(entry)

        # Keep only recent entries
        if len(self._recent_logs) > self._max_log_entries:
            self._recent_logs = self._recent_logs[-self._max_log_entries :]

        # Record log metric
        if self._log_counter:
            self._log_counter.add(1, {"level": entry.level})

    def get_recent_logs(self, limit: int = 100) -> List[LogEntry]:
        """Get recent log entries."""
        if limit <= 0 or limit > len(self._recent_logs):
            limit = len(self._recent_logs)
        return self._recent_logs[-limit:]

    @contextmanager
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Start a new trace span."""
        if self.tracer:
            with self.tracer.start_as_current_span(name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                yield span
        else:
            yield None

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for API response."""
        uptime_seconds = time.time() - self._start_time

        # Calculate average throughput
        with self._throughput_lock:
            total_requests = self._request_count_per_minute
        avg_throughput = total_requests / max(
            1, uptime_seconds / 60
        )  # requests per minute

        return {
            "service": self.config.service_name,
            "version": self.config.service_version,
            "environment": self.config.environment,
            "uptime_seconds": round(uptime_seconds, 1),
            "log_count": len(self._recent_logs),
            "otel_endpoint": self.config.get_otlp_endpoint() or "none",
            "total_requests": total_requests,
            "throughput": {
                "requests_per_minute": round(avg_throughput, 2),
                "current_rpm": self._resource_snapshot.get("throughput_rpm", 0),
            },
            "resources": {
                "cpu_percent": psutil.cpu_percent(interval=None)
                if PSUTIL_AVAILABLE
                else 0,
                "memory_percent": psutil.virtual_memory().percent
                if PSUTIL_AVAILABLE
                else 0,
                "memory_used_mb": int(psutil.virtual_memory().used / (1024 * 1024))
                if PSUTIL_AVAILABLE
                else 0,
                "disk_percent": psutil.disk_usage("/").percent
                if PSUTIL_AVAILABLE
                else 0,
                "disk_used_gb": round(
                    psutil.disk_usage("/").used / (1024 * 1024 * 1024), 2
                )
                if PSUTIL_AVAILABLE
                else 0,
            },
            "psutil_available": PSUTIL_AVAILABLE,
        }

    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        snapshot = {"timestamp": datetime.utcnow().isoformat()}
        if PSUTIL_AVAILABLE:
            snapshot.update(
                {
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "memory_percent": psutil.virtual_memory().percent,
                    "memory_used_mb": int(psutil.virtual_memory().used / (1024 * 1024)),
                    "disk_percent": psutil.disk_usage("/").percent,
                    "disk_used_gb": round(
                        psutil.disk_usage("/").used / (1024 * 1024 * 1024), 2
                    ),
                }
            )
        else:
            snapshot.update(
                {
                    "cpu_percent": 0.0,
                    "memory_percent": 0.0,
                    "memory_used_mb": 0,
                    "disk_percent": 0.0,
                    "disk_used_gb": 0,
                }
            )
        snapshot["psutil_available"] = PSUTIL_AVAILABLE
        return snapshot

    def get_logs_response(self, limit: int = 100) -> Dict[str, Any]:
        """Get logs for API response."""
        logs = self.get_recent_logs(limit)
        return {
            "service": self.config.service_name,
            "total": len(logs),
            "logs": [
                {
                    "timestamp": log.timestamp,
                    "level": log.level,
                    "message": log.message,
                    "trace_id": log.trace_id,
                    "span_id": log.span_id,
                    "attributes": log.attributes,
                }
                for log in logs
            ],
        }


class OTelLogHandler(logging.Handler):
    """Custom log handler that captures logs for OTel."""

    def __init__(self, telemetry: Telemetry):
        super().__init__()
        self.telemetry = telemetry

    def emit(self, record: logging.LogRecord) -> None:
        """Process a log record."""
        try:
            # Get trace context
            span = trace.get_current_span()
            span_context = span.get_span_context() if span else None

            entry = LogEntry(
                timestamp=datetime.utcnow().isoformat() + "Z",
                level=record.levelname,
                message=self.format(record),
                service=self.telemetry.config.service_name,
                trace_id=format(span_context.trace_id, "032x")
                if span_context and span_context.is_valid
                else None,
                span_id=format(span_context.span_id, "016x")
                if span_context and span_context.is_valid
                else None,
                attributes={
                    "logger": record.name,
                    "filename": record.filename,
                    "lineno": record.lineno,
                },
            )

            self.telemetry.add_log_entry(entry)
        except Exception:
            pass  # Don't let logging errors crash the app


# Global telemetry instance
_telemetry: Optional[Telemetry] = None


def get_telemetry() -> Optional[Telemetry]:
    """Get the global telemetry instance."""
    return _telemetry


def init_telemetry(app: FastAPI, config: Optional[TelemetryConfig] = None) -> Telemetry:
    """Initialize global telemetry."""
    global _telemetry

    if config is None:
        # Load from environment or use defaults
        config = TelemetryConfig(
            service_name=os.getenv("OTEL_SERVICE_NAME", "orange3-web-backend"),
            service_version=os.getenv("SERVICE_VERSION", _get_version()),
            environment=os.getenv("ENVIRONMENT", "development"),
            otel_endpoint=os.getenv("OTEL_ENDPOINT"),
            enable_console=os.getenv("OTEL_CONSOLE", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    _telemetry = Telemetry(config)
    _telemetry.setup(app)
    return _telemetry
