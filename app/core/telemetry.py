"""
OpenTelemetry integration for Orange3-Web Backend.

Provides:
- Tracing: Distributed tracing with span creation
- Metrics: Request counts, latency, custom metrics
- Logging: Structured logging with trace context
"""
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.trace import Status, StatusCode

# Conditional OTLP imports
try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False


@dataclass
class TelemetryConfig:
    """Configuration for OpenTelemetry."""
    service_name: str = "orange3-web-backend"
    service_version: str = "0.27.5"
    environment: str = "development"
    otel_endpoint: Optional[str] = None
    enable_console: bool = False
    log_level: str = "INFO"
    metric_interval_ms: int = 15000


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
        
        # Backend metrics tracking
        self._metrics: Dict[str, Dict] = {}
        
    def setup(self, app: FastAPI) -> None:
        """Initialize OpenTelemetry with the FastAPI app."""
        # Create resource
        resource = Resource.create({
            SERVICE_NAME: self.config.service_name,
            SERVICE_VERSION: self.config.service_version,
            "environment": self.config.environment,
        })
        
        # Setup tracing
        self._setup_tracing(resource)
        
        # Setup metrics
        self._setup_metrics(resource)
        
        # Setup logging instrumentation
        self._setup_logging()
        
        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)
        
        logging.info(f"[OTel] Telemetry initialized for service: {self.config.service_name}")
        
    def _setup_tracing(self, resource: Resource) -> None:
        """Configure trace provider."""
        provider = TracerProvider(resource=resource)
        
        if self.config.otel_endpoint and OTLP_AVAILABLE:
            # OTLP exporter
            exporter = OTLPSpanExporter(
                endpoint=f"http://{self.config.otel_endpoint}/v1/traces"
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logging.info(f"[OTel] Tracing: OTLP exporter -> {self.config.otel_endpoint}")
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
        
        if self.config.otel_endpoint and OTLP_AVAILABLE:
            # OTLP exporter
            exporter = OTLPMetricExporter(
                endpoint=f"http://{self.config.otel_endpoint}/v1/metrics"
            )
            readers.append(PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=self.config.metric_interval_ms
            ))
            logging.info(f"[OTel] Metrics: OTLP exporter -> {self.config.otel_endpoint}")
        elif self.config.enable_console:
            # Console exporter for debugging
            readers.append(PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=self.config.metric_interval_ms
            ))
            logging.info("[OTel] Metrics: Console exporter")
        
        if readers:
            provider = MeterProvider(resource=resource, metric_readers=readers)
        else:
            provider = MeterProvider(resource=resource)
        metrics.set_meter_provider(provider)
        self.meter = metrics.get_meter(self.config.service_name)
        
        # Create metrics instruments
        self._request_counter = self.meter.create_counter(
            "http.requests.total",
            description="Total HTTP requests",
            unit="{requests}"
        )
        
        self._request_duration = self.meter.create_histogram(
            "http.request.duration",
            description="HTTP request duration in milliseconds",
            unit="ms"
        )
        
        self._log_counter = self.meter.create_counter(
            "logs.total",
            description="Total log entries",
            unit="{logs}"
        )
        
    def _setup_logging(self) -> None:
        """Configure logging instrumentation."""
        # Instrument Python logging to include trace context
        try:
            LoggingInstrumentor().instrument(set_logging_format=True)
        except Exception as e:
            logging.warning(f"LoggingInstrumentor failed: {e}")
        
        # Setup custom log handler
        handler = OTelLogHandler(self)
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        handler.setLevel(log_level)
        handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
        
        # Add handler to root logger and ensure level is set
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(handler)
        
        # Also add to uvicorn loggers
        for logger_name in ['uvicorn', 'uvicorn.access', 'uvicorn.error', 'fastapi']:
            logger = logging.getLogger(logger_name)
            logger.addHandler(handler)
        
        logging.info(f"[OTel] Logging initialized with level: {self.config.log_level}")
        
    def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        error: Optional[str] = None
    ) -> None:
        """Record HTTP request metrics."""
        status = "success" if status_code < 400 else ("client_error" if status_code < 500 else "error")
        
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
            
    def add_log_entry(self, entry: LogEntry) -> None:
        """Add a log entry to the buffer."""
        self._recent_logs.append(entry)
        
        # Keep only recent entries
        if len(self._recent_logs) > self._max_log_entries:
            self._recent_logs = self._recent_logs[-self._max_log_entries:]
            
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
        return {
            "service": self.config.service_name,
            "version": self.config.service_version,
            "environment": self.config.environment,
            "uptime": f"{time.time() - self._start_time:.1f}s",
            "log_count": len(self._recent_logs),
            "otel_endpoint": self.config.otel_endpoint or "none",
        }
    
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
            ]
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
                trace_id=format(span_context.trace_id, '032x') if span_context and span_context.is_valid else None,
                span_id=format(span_context.span_id, '016x') if span_context and span_context.is_valid else None,
                attributes={
                    "logger": record.name,
                    "filename": record.filename,
                    "lineno": record.lineno,
                }
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
            service_version=os.getenv("SERVICE_VERSION", "0.27.5"),
            environment=os.getenv("ENVIRONMENT", "development"),
            otel_endpoint=os.getenv("OTEL_ENDPOINT"),
            enable_console=os.getenv("OTEL_CONSOLE", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
    
    _telemetry = Telemetry(config)
    _telemetry.setup(app)
    return _telemetry

