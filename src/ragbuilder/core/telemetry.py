import os
import uuid
import platform
from pathlib import Path
from platformdirs import user_data_dir
from typing import Union, Dict, Any, Literal
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider, Counter
from opentelemetry.sdk.metrics.export import AggregationTemporality
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource
from ragbuilder._version import __version__
from ragbuilder.core.results import DataIngestResults, RetrievalResults, GenerationResults
import logging
from contextlib import contextmanager
from datetime import datetime
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
ModuleType = Literal["data_ingest", "retriever", "generation", "ragbuilder"]

class RAGBuilderTelemetry:
    def __init__(self):
        self.enabled = False  # Start disabled by default
        self.tracer = None
        self.meter = None
        self.meter_provider = None
        
        try:
            self.enabled = os.getenv("ENABLE_ANALYTICS", "true").lower() != "false"
            self.user_id = self._get_or_create_user_id()
            
            if self.enabled:
                self._setup_telemetry()
        except Exception as e:
            logger.debug(f"Failed to initialize telemetry: {e}")
            self.enabled = False
    
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
            logger.debug(f"Failed to persist user ID: {e}")
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
                "os.machine": platform.machine(),
                "anonymous_user_id": self.user_id
            })

            # Setup tracing
            trace_provider = TracerProvider(resource=resource)
            trace_exporter = OTLPSpanExporter(
                endpoint="https://api.honeycomb.io/v1/traces",
                headers={"x-honeycomb-team": os.getenv("HONEYCOMB_API_KEY", "hcaik_01jfmbeze0tvzt6rwv3dvcc276addbpctde4kyv3bq2f7zbng30fqjysq7")}
            )
            trace_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
            trace.set_tracer_provider(trace_provider)
            self.tracer = trace.get_tracer(__name__)

            # Setup metrics
            metric_exporter = OTLPMetricExporter(
                endpoint="https://api.honeycomb.io/v1/metrics",
                headers={
                    "x-honeycomb-team": os.getenv("HONEYCOMB_API_KEY", "hcaik_01jfmbeze0tvzt6rwv3dvcc276addbpctde4kyv3bq2f7zbng30fqjysq7"),
                    "x-honeycomb-dataset": "ragbuilder-metrics"
                },
                preferred_temporality={Counter: AggregationTemporality.DELTA}
            )
            reader = PeriodicExportingMetricReader(metric_exporter)
            self.meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(self.meter_provider)
            self.meter = metrics.get_meter(__name__)

            self._setup_metrics()

        except Exception as e:
            self.enabled = False
            logger.debug(f"Failed to initialize telemetry: {e}")

    def _setup_metrics(self):
        """Initialize metrics instruments."""
        try:
            self.optimization_starts = self.meter.create_counter(
                "optimization.starts",
                description="Number of optimization runs started"
            )
            self.optimization_completions = self.meter.create_counter(
                "optimization.completions",
                description="Number of successful optimization completions"
            )
            self.errors = self.meter.create_counter(
                "errors",
                description="Number of errors by type"
            )
            self.eval_datagen_starts = self.meter.create_counter(
                "eval_datagen.starts",
                description="Number of evaluation dataset generations started"
            )
            self.eval_datagen_completions = self.meter.create_counter(
                "eval_datagen.completions",
                description="Number of successful evaluation dataset generations"
            )
        except Exception as e:
            logger.debug(f"Failed to setup metrics: {e}")
            self.enabled = False

    def _safe_add_counter(self, counter, value: int = 1, attributes: Dict = None):
        """Safely add to counter with error handling."""
        try:
            if self.enabled and counter:
                counter.add(value, attributes)
        except Exception as e:
            logger.debug(f"Failed to add to counter: {e}")

    def _safe_set_attribute(self, span, key: str, value: Any):
        """Safely set span attribute with error handling."""
        try:
            if span:
                span.set_attribute(key, value)
        except Exception as e:
            logger.debug(f"Failed to set span attribute {key}: {e}")

    @contextmanager
    def eval_datagen_span(self, **attributes):
        """Context manager for eval data generation spans."""
        if not self.enabled or not self.tracer:
            yield None
            return
            
        span = None
        start_time = datetime.now()
        
        try:
            with self.tracer.start_as_current_span("eval_data_generation") as span:
                self._safe_add_counter(self.eval_datagen_starts, 1)
                self._safe_set_attribute(span, "user_id", self.user_id)
                
                # Add all provided attributes
                for key, value in attributes.items():
                    if value is not None:
                        self._safe_set_attribute(span, key, value)
                        
                try:
                    yield span
                    
                    self._safe_add_counter(self.eval_datagen_completions, 1)
                    duration = (datetime.now() - start_time).total_seconds()
                    self._safe_set_attribute(span, "generation_duration", duration)
                    self._safe_set_attribute(span, "generation_success", True)
                    
                except Exception as e:
                    self._safe_set_attribute(span, "generation_success", False)
                    self._safe_set_attribute(span, "error_type", e.__class__.__name__)
                    self._safe_set_attribute(span, "error_message", str(e))
                    raise e
                    
        except Exception as e:
            logger.debug(f"Error in eval data generation span: {e}")
            if span is None:
                yield None
            raise

    @contextmanager
    def optimization_span(self, module: ModuleType, config: Dict[str, Any]):
        """Context manager for optimization spans with module-specific attributes."""
        if not self.enabled or not self.tracer:
            yield None
            return
        
        span = None
        start_time = datetime.now()
        
        try:
            with self.tracer.start_as_current_span(f"{module}_optimization") as span:
                self._safe_add_counter(self.optimization_starts, 1, {"module": module})
                self._safe_set_attribute(span, "module", module)
                self._safe_set_attribute(span, "user_id", self.user_id)

                if module != "ragbuilder":
                    self._safe_set_attribute(span, "n_trials", config.get("optimization", {}).get("n_trials", 0))
                    self._safe_set_attribute(span, "n_jobs", config.get("optimization", {}).get("n_jobs", 1))
                    self._safe_set_attribute(span, "is_default_config", config.get("metadata", {}).get("is_default", False))

                # Module-specific attributes
                try:
                    if module == "data_ingest":
                        self._set_data_ingest_attributes(span, config)
                    elif module == "retriever":
                        self._set_retriever_attributes(span, config)
                    elif module == "generation":
                        self._set_generator_attributes(span, config)
                except Exception:
                    logger.debug(f"Failed to set telemetry span attributes for module {module}: {e}")

                try:
                    yield span
                    
                    self._safe_add_counter(self.optimization_completions, 1, {"module": module})
                    duration = (datetime.now() - start_time).total_seconds()
                    self._safe_set_attribute(span, "optimization_duration", duration)
                    self._safe_set_attribute(span, "optimization_success", True)
                    
                except Exception as e:
                    self._safe_set_attribute(span, "optimization_success", False)
                    self._safe_set_attribute(span, "error_type", e.__class__.__name__)
                    self._safe_set_attribute(span, "error_message", str(e))
                    raise  
        
        except Exception as e:
            logger.debug(f"Error in optimization span: {e}")
            if span is None:
                yield None
            raise 


    def _set_data_ingest_attributes(self, span, config):
        """Set data ingestion specific span attributes."""
        input_source = config.get("input_source", "")
        input_source_type = "url" if urlparse(input_source).scheme in ['http', 'https'] else (
            'dir' if os.path.isdir(input_source) else 'file'
        )
        self._safe_set_attribute(span, "input_source_type", input_source_type)
        config.get("sampling_rate") and self._safe_set_attribute(span, "sampling_rate", config.get("sampling_rate"))
        config.get("graph") and self._safe_set_attribute(span, "graph_enabled", True)

    def _set_retriever_attributes(self, span, config):
        """Set retriever specific span attributes."""
        retrievers = config.get("retrievers", [])
        self._safe_set_attribute(span, "n_retrievers", len(retrievers))
        self._safe_set_attribute(span, "reranker_enabled", bool(config.get("rerankers", [])))

    def _set_generator_attributes(self, span, config):
        """Set generator specific span attributes."""
        self._safe_set_attribute(span, "n_llms", len(config.get("llms", [])))

    def update_optimization_results(self, span, results: Union[DataIngestResults, RetrievalResults, GenerationResults], module: ModuleType):
        """Update span with optimization results."""
        if not self.enabled or not span:
            return

        self._safe_set_attribute(span, "best_score", results.best_score)
        if results.avg_latency is not None:
            self._safe_set_attribute(span, "avg_latency", results.avg_latency)
        if results.error_rate is not None:
            self._safe_set_attribute(span, "error_rate", results.error_rate)
        
        if module == "data_ingest" and isinstance(results, DataIngestResults):
            config = results.get_config_summary()
            
            self._safe_set_attribute(span, "best_parser_type", config["document_loader"])
            self._safe_set_attribute(span, "best_embedding_model", config["embedding_model"])
            self._safe_set_attribute(span, "best_chunking_strategy", config["chunking_strategy"])
            self._safe_set_attribute(span, "best_chunk_size", config["chunk_size"])
            self._safe_set_attribute(span, "best_chunk_overlap", config["chunk_overlap"])
            self._safe_set_attribute(span, "vector_db", config["vector_database"])
            self._safe_set_attribute(span, "data_source_size", getattr(results.best_pipeline, "data_source_size", 0))
            
        elif module == "retriever" and isinstance(results, RetrievalResults):
            config = results.get_config_summary()
            self._safe_set_attribute(span, "best_retriever_types", config["retrievers"])
            if config.get("rerankers"):
                self._safe_set_attribute(span, "best_reranker", config["rerankers"])
            self._safe_set_attribute(span, "best_top_k", config.get("top_k", ""))

        elif module == "generation" and isinstance(results, GenerationResults):
            config_summary = results.get_config_summary()
            self._safe_set_attribute(span, "model", config_summary["model"])
            self._safe_set_attribute(span, "temperature", config_summary["temperature"])            

    def track_error(self, module: ModuleType, error: Exception, context: Dict[str, Any]):
        """Track error occurrence with span and metrics."""
        if not self.enabled:
            return

        with self.tracer.start_as_current_span(f"{module}_error") as span:
            self._safe_set_attribute(span, "module", module)
            self._safe_set_attribute(span, "user_id", self.user_id)
            self._safe_set_attribute(span, "error_type", error.__class__.__name__)
            self._safe_set_attribute(span, "error_message", str(error))
            
            for key, value in context.items():
                if isinstance(value, (str, int, float, bool)):
                    self._safe_set_attribute(span, f"context.{key}", value)

        self._safe_add_counter(self.errors, 1, {"module": module, "error_type": error.__class__.__name__})

    def flush(self):
        try:
            if self.enabled and self.meter_provider:
                self.meter_provider.force_flush()
        except Exception as e:
            logger.debug(f"Error flushing telemetry: {e}")

    def shutdown(self):
        if self.enabled and self.meter_provider:
            self.meter_provider.force_flush()
            self.meter_provider.shutdown()

telemetry = RAGBuilderTelemetry()