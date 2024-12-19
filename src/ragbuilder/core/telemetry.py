import os
import uuid
import platform
from pathlib import Path
from platformdirs import user_data_dir
from typing import Optional, Dict, Any, Literal
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource
from ragbuilder._version import __version__
import logging
from contextlib import contextmanager
from datetime import datetime
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
ModuleType = Literal["data_ingest", "retriever", "generator"]

class RAGBuilderTelemetry:
    def __init__(self):
        self.enabled = os.getenv("ENABLE_ANALYTICS", "true").lower() != "false"
        self.user_id = self._get_or_create_user_id()
        
        if self.enabled:
            self._setup_telemetry()
    
    def _get_or_create_user_id(self) -> str:
        try:
            data_dir = Path(user_data_dir(appname="ragbuilder"))
            id_file = data_dir / "uuid"
            if id_file.exists():
                return id_file.read_text().strip()

            user_id = 'a-' + uuid.uuid4().hex
            data_dir.mkdir(parents=True, exist_ok=True)        
            id_file.write_text(user_id)
            return user_id
        except Exception as e:
            logger.warning(f"Failed to persist user ID: {e}")
            return 'a-' + uuid.uuid4().hex

    def _setup_telemetry(self):
        """Initialize OpenTelemetry with tracer and meter."""
        try:
            resource = Resource.create({
                "service.name": "ragbuilder",
                "service.version": __version__,
                "environment": os.getenv("RAGBUILDER_ENV", "production"),
                "python.version": platform.python_version(),
                "os.type": platform.system(),
                "anonymous_user_id": self.user_id
            })

            # Setup tracing
            trace_provider = TracerProvider(resource=resource)
            trace_exporter = OTLPSpanExporter(
                endpoint="https://api.honeycomb.io/v1/traces",
                headers={"x-honeycomb-team": os.getenv("HONEYCOMB_API_KEY", "CM7BwTAsHffuxBeRgyJpkN")}
            )
            trace_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
            trace.set_tracer_provider(trace_provider)
            self.tracer = trace.get_tracer(__name__)

            # Setup metrics
            metric_exporter = OTLPMetricExporter(
                endpoint="https://api.honeycomb.io/v1/metrics",
                headers={"x-honeycomb-team": os.getenv("HONEYCOMB_API_KEY", "CM7BwTAsHffuxBeRgyJpkN")}
            )
            reader = PeriodicExportingMetricReader(metric_exporter)
            meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(meter_provider)
            self.meter = metrics.get_meter(__name__)

            self._setup_metrics()

        except Exception as e:
            self.enabled = False
            logger.warning(f"Failed to initialize telemetry: {e}")

    def _setup_metrics(self):
        """Initialize metrics instruments."""
        # User tracking metrics
        self.users_total = self.meter.create_up_down_counter(
            "ragbuilder.users.total",
            description="Total number of users"
        )
        self.users_by_module = self.meter.create_counter(
            "ragbuilder.users.module",
            description="Users by module"
        )

        # Core operation metrics
        self.optimization_starts = self.meter.create_counter(
            "ragbuilder.optimization.starts",
            description="Number of optimization runs started"
        )
        self.optimization_completions = self.meter.create_counter(
            "ragbuilder.optimization.completions",
            description="Number of successful optimization completions"
        )

        # Error metrics
        self.errors = self.meter.create_counter(
            "ragbuilder.errors",
            description="Number of errors by type"
        )

        # Record initial user
        self.users_total.add(1)

    @contextmanager
    def optimization_span(self, module: ModuleType, config: Dict[str, Any]):
        """Context manager for optimization spans with module-specific attributes."""
        if not self.enabled:
            yield None
            return

        self.optimization_starts.add(1, {"module": module})
        self.users_by_module.add(1, {"module": module})
        start_time = datetime.now()
        
        with self.tracer.start_as_current_span(f"{module}_optimization") as span:
            span.set_attribute("module", module)
            span.set_attribute("user_id", self.user_id)
            span.set_attribute("n_trials", config.get("optimization", {}).get("n_trials", 0))
            span.set_attribute("n_jobs", config.get("optimization", {}).get("n_jobs", 1))
            span.set_attribute("is_default_config", config.get("metadata", {}).get("is_default", False))

            # Module-specific attributes
            if module == "data_ingest":
                self._set_data_ingest_attributes(span, config)
            elif module == "retriever":
                self._set_retriever_attributes(span, config)
            elif module == "generator":
                self._set_generator_attributes(span, config)

            try:
                yield span
                
                self.optimization_completions.add(1, {"module": module})
                
                # Add duration
                duration = (datetime.now() - start_time).total_seconds()
                span.set_attribute("optimization_duration", duration)
                
            except Exception as e:
                span.set_attribute("optimization_success", False)
                raise e
            else:
                span.set_attribute("optimization_success", True)

    def _set_data_ingest_attributes(self, span, config):
        """Set data ingestion specific span attributes."""
        input_source = config.get("input_source", "")
        input_source_type = "url" if urlparse(input_source).scheme in ['http', 'https'] else (
            'dir' if os.path.isdir(input_source) else 'file'
        )
        span.set_attribute("input_source_type", input_source_type)
        config.get("sampling_rate") and span.set_attribute("sampling_rate", config.get("sampling_rate"))
        config.get("graph") and span.set_attribute("graph_enabled", True)

    def _set_retriever_attributes(self, span, config):
        """Set retriever specific span attributes."""
        retrievers = config.get("retrievers", [])
        span.set_attribute("n_retrievers", len(retrievers))
        span.set_attribute("reranker_enabled", bool(config.get("rerankers", [])))

    def _set_generator_attributes(self, span, config):
        """Set generator specific span attributes."""
        span.set_attribute("n_llms", len(config.get("llms", [])))

    def update_optimization_results(self, span, results: Dict[str, Any], module: ModuleType):
        """Update span with optimization results."""
        if not self.enabled or not span:
            return

        span.set_attribute("best_score", results.get("best_score", 0))
        
        if module == "data_ingest" and "best_config" in results:
            config = results["best_config"]
            span.set_attribute("best_parser_type", str(config.document_loader.type))
            model_name = config.embedding_model.model_kwargs.get('model') or config.embedding_model.model_kwargs.get('model_name', '')
            embedding_model = f"{config.embedding_model.type}:{model_name}" if model_name else str(config.embedding_model.type)
            span.set_attribute("best_embedding_model", embedding_model)
            span.set_attribute("best_chunking_strategy", str(config.chunking_strategy.type))
            span.set_attribute("best_chunk_size", config.chunk_size)
            span.set_attribute("best_chunk_overlap", config.chunk_overlap)
            span.set_attribute("vector_db", str(config.vector_database.type))
            span.set_attribute("data_source_size", getattr(results["best_pipeline"], "data_source_size", 0))
            
        elif module == "retriever" and "best_config" in results:
            config = results["best_config"]
            span.set_attribute("best_retriever_types", [str(r.type) for r in config.retrievers])
            if config.rerankers:
                span.set_attribute("best_reranker", str(config.rerankers[0].type))
            span.set_attribute("best_top_k", config.top_k)

    def track_error(self, module: ModuleType, error: Exception, context: Dict[str, Any]):
        """Track error occurrence with span and metrics."""
        if not self.enabled:
            return

        with self.tracer.start_as_current_span(f"{module}_error") as span:
            span.set_attribute("module", module)
            span.set_attribute("user_id", self.user_id)
            span.set_attribute("error_type", error.__class__.__name__)
            span.set_attribute("error_message", str(error))
            
            for key, value in context.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(f"context.{key}", value)

        self.errors.add(1, {"module": module, "error_type": error.__class__.__name__})

telemetry = RAGBuilderTelemetry()