from typing import Optional, Dict, Any
import requests
import logging
from optuna.study import Study
from optuna.trial import FrozenTrial
from .config import DataIngestOptionsConfig

logger = logging.getLogger(__name__)

class UICallback:
    """Callback to integrate optimization with UI visualization."""
    
    def __init__(self, 
                 study_name: str,
                 config: DataIngestOptionsConfig,
                 server_url: str = "http://localhost:8005",
                 enable_ui: bool = True):
        """
        Args:
            study_name: Name of the optimization study
            config: Configuration for data ingestion optimization
            server_url: URL of the UI server
            enable_ui: Whether to enable UI integration
        """
        self.study_name = study_name
        self.config = config
        self.server_url = server_url.rstrip('/')
        self.enable_ui = enable_ui
        self.run_id: Optional[int] = None
        
        if self.enable_ui:
            self.run_id = self._register_study()

    def _register_study(self) -> Optional[int]:
        """Register new optimization study with UI server."""
        try:
            response = requests.post(
                f"{self.server_url}/api/data_ingest/register_study",
                json={
                    "study_name": self.study_name,
                    "config": self.config.model_dump()
                }
            )
            response.raise_for_status()
            return response.json()["run_id"]
        except Exception as e:
            logger.warning(f"Failed to register study with UI: {e}")
            return None

    def _log_trial(self, trial: FrozenTrial, metrics: Dict[str, Any]):
        """Log trial results to UI server."""
        if not self.run_id:
            return

        try:
            requests.post(
                f"{self.server_url}/api/data_ingest/log_trial",
                json={
                    "run_id": self.run_id,
                    "trial_number": trial.number,
                    "config": trial.params,
                    "metrics": metrics
                }
            )
        except Exception as e:
            logger.warning(f"Failed to log trial results to UI: {e}")

    def __call__(self, study: Study, trial: FrozenTrial):
        """Called after each trial completion."""
        if not self.enable_ui:
            return

        metrics = {
            "retrieval_score": trial.value,
            # Add additional metrics as needed
            "state": trial.state.name,
            "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
            "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        }
        
        self._log_trial(trial, metrics) 