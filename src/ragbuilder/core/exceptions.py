
class RAGBuilderError(Exception):
    """Base exception for RAGBuilder."""
    pass

class DependencyError(RAGBuilderError):
    """Raised when a required dependency is not met."""
    pass

class ConfigurationError(RAGBuilderError):
    """Invalid configuration error."""
    pass

class EnvironmentError(RAGBuilderError):
    """Missing environment variables error."""
    pass

class PipelineError(RAGBuilderError):
    """Pipeline execution error."""
    pass

class ComponentError(RAGBuilderError):
    """Component initialization or execution error."""
    pass

class OptimizationError(RAGBuilderError):
    """Optimization error."""
    pass

class EvaluationError(RAGBuilderError):
    """Evaluation error."""
    pass