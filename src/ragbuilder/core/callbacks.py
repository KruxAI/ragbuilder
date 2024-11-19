import sqlite3
import json
import time
import logging
import random
from typing import Dict, Any, Optional, Protocol
from datetime import datetime
from contextlib import contextmanager
from optuna.study import Study
from optuna.trial import Trial
from ragbuilder.config.data_ingest import DataIngestOptionsConfig

logger = logging.getLogger(__name__)


class DBLoggerCallback(Protocol):
    """Callback to log optimization results to the DB for optional UI visualization."""
    
    def __init__(self, 
                 study_name: str,
                 config: DataIngestOptionsConfig):
        """
        Args:
            study_name: Name of the optimization study
            config: Configuration for data ingestion optimization
        """
        self.study_name = study_name
        self.config = config
        self.db_path = config.database_path
        self.log_path = logger.handlers[0].baseFilename if logger.handlers else None
        self.run_id = None

        try:
            self._init_tables()
            self.run_id = self._create_run()
            logger.info(f"Initialized DB logging for run_id: {self.run_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize database logging: {e}")

            
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()    

    def _init_tables(self):
        """Initialize database tables if they don't exist."""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS run_details (
                run_id              BIGINT PRIMARY KEY,
                module_type         VARCHAR(100),
                status              VARCHAR(30),
                description         TEXT,
                src_data            TEXT,
                log_path            TEXT,
                run_config          JSON,
                disabled_opts       JSON,
                run_ts              BIGINT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS data_ingest_eval_summary (
                eval_id                 BIGINT PRIMARY KEY,
                run_id                  BIGINT,
                trial_number            INTEGER,
                config                  JSON,
                avg_score               FLOAT,
                avg_latency             FLOAT,
                total_questions         INTEGER,
                successful_questions    INTEGER,
                error_rate              FLOAT,
                eval_start_ts           BIGINT,
                eval_end_ts             BIGINT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS data_ingest_eval_details (
                eval_id             BIGINT,
                question_id         BIGINT,
                question            TEXT,
                retrieved_chunks    JSON,
                relevance_scores    JSON,
                weighted_score      FLOAT,
                latency             FLOAT,
                error               TEXT,
                eval_ts             BIGINT
            )
            """
        ]
        
        with self._get_connection() as conn:
            for table in tables:
                conn.execute(table)
            conn.commit()
            logger.debug("Database tables initialized")

    def _create_run(self) -> int:
        """Create a new run entry and return the run_id."""
        try:
            with self._get_connection() as conn:
                run_id = int(time.time())
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO run_details (
                        run_id,
                        module_type,
                        status,
                        description,
                        src_data,
                        log_path,
                        run_config,
                        disabled_opts,
                        run_ts
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        "data_ingest",
                        "Running",
                        self.study_name,
                        self.config.input_source,
                        self.log_path,
                        json.dumps(self.config.model_dump()),
                        None,
                        run_id
                    )
                )
                conn.commit()
                return run_id
                
        except Exception as e:
            logger.error(f"Failed to create run entry: {e}")
            raise

    def _update_run_status(self, status: str):
        """Update the status of a run."""
        if not self.run_id:
            return
        
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE run_details SET status = ? WHERE run_id = ?",
                    (status, self.run_id)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update run status: {e}")

    def _log_trial(self, trial: Trial, results: Dict[str, Any]) -> Optional[int]:
        """Log trial results to database."""
        try:
            eval_id = int(time.time()*1000+random.randint(1, 1000))
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN")
                try:
                    # Log Summary
                    cursor.execute(
                        """
                        INSERT INTO data_ingest_eval_summary (
                            eval_id,
                            run_id,
                            trial_number,
                            config,
                            avg_score,
                            avg_latency,
                            total_questions,
                            successful_questions,
                            error_rate,
                            eval_start_ts,
                            eval_end_ts
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            eval_id,
                            self.run_id,
                            trial.number,
                            json.dumps(results['config']),
                            results['avg_score'],
                            results['metrics']['avg_latency'],
                            results['metrics']['total_questions'],
                            results['metrics']['successful_questions'],
                            results['metrics']['error_rate'],
                            int(trial.datetime_start.timestamp() * 1000) if hasattr(trial, 'datetime_start') else None,
                            int(time.time()*1000)
                        )
                    )
                    
                    # Log details 
                    cursor.executemany(
                        """
                        INSERT INTO data_ingest_eval_details (
                            eval_id,
                            question_id,
                            question,
                            retrieved_chunks,
                            relevance_scores,
                            weighted_score,
                            latency,
                            error,
                            eval_ts
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                eval_id,
                                idx,
                                detail['question'],
                                json.dumps(detail.get('retrieved_chunks', [])),
                                json.dumps(detail.get('relevance_scores', [])),
                                detail.get('weighted_score'),
                                detail.get('latency'),
                                detail.get('error'),
                                detail['eval_timestamp']
                            )
                            for idx, detail in enumerate(results['question_details'])
                        ]
                    )
                    
                    conn.commit()
                    return eval_id
                
                except Exception as e:
                    cursor.execute("ROLLBACK")
                    raise e
                
        except Exception as e:
            logger.error(f"Failed to log trial results: {e}")
            return None

    def __call__(self, study: Study, trial: Trial):
        """Called after each trial completion."""
        results = study.user_attrs.get(f"trial_{trial.number}_results", {})
        if results:
            self._log_trial(trial, results)
            logger.debug(f"Logged results for trial {trial.number}")

    def __del__(self):
        """Update run status when optimization completes."""
        self._update_run_status("Success")