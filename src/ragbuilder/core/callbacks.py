import sqlite3
import json
import time
import logging
import random
from typing import Dict, Any, Optional, Protocol, Union
from datetime import datetime
from contextlib import contextmanager
from optuna.study import Study
from optuna.trial import Trial
from ragbuilder.config.data_ingest import DataIngestOptionsConfig
from ragbuilder.config.retriever import RetrievalOptionsConfig
from ragbuilder.config.generation import GenerationOptionsConfig
from .utils import serialize_config
from datasets import Dataset
logger = logging.getLogger(__name__)


class DBLoggerCallback(Protocol):
    """Callback to log optimization results to the DB for optional UI visualization."""
    
    def __init__(self, 
                 study_name: str,
                 config: Union[DataIngestOptionsConfig, RetrievalOptionsConfig, GenerationOptionsConfig],
                 module_type: str):
        """
        Args:
            study_name: Name of the optimization study
            config: Configuration for optimization
            module_type: Type of module ('data_ingest' or 'retriever' or 'generation')
        """
        self.study_name = study_name
        self.config = config
        self.module_type = module_type
        self.db_path = config.database_path
        self.log_path = logger.handlers[0].baseFilename if logger.handlers else None
        self.run_id = None

        try:
            self._init_tables()
            self.run_id = self._create_run()
            logger.debug(f"Initialized DB logging for run_id: {self.run_id}")
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
            """,
            """
            CREATE TABLE IF NOT EXISTS retriever_eval_summary (
                eval_id                 BIGINT PRIMARY KEY,
                run_id                  BIGINT,
                trial_number            INTEGER,
                config                  JSON,
                avg_score               FLOAT,
                avg_context_precision   FLOAT,
                avg_context_recall      FLOAT,
                avg_latency             FLOAT,
                total_questions         INTEGER,
                successful_questions    INTEGER,
                error_rate              FLOAT,
                eval_start_ts           BIGINT,
                eval_end_ts             BIGINT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS retriever_eval_details (
                eval_id             BIGINT,
                question_id         BIGINT,
                question            TEXT,
                contexts           JSON,
                ground_truth       TEXT,
                context_precision  FLOAT,
                context_recall     FLOAT,
                f1_score           FLOAT,
                latency           FLOAT,
                error             TEXT,
                eval_ts           BIGINT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS generation_eval_details (
                eval_id             BIGINT,
                question_id         BIGINT,
                question            TEXT,
                answer              TEXT,
                ground_truth        TEXT,
                prompt_key          TEXT,
                prompt              TEXT,
                answer_correctness  FLOAT
            )
            """,
                        """
            CREATE TABLE IF NOT EXISTS generation_eval_summary (
                run_id              BIGINT,
                eval_id             BIGINT,
                prompt_key          BIGINT,
                prompt              TEXT,
                config              TEXT,
                average_correctness FLOAT
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
                    """INSERT INTO run_details (
                        run_id, module_type, status, description,
                        src_data, log_path, run_config, disabled_opts, run_ts
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        self.module_type,
                        "Running",
                        self.study_name,
                        getattr(self.config, 'input_source', None),
                        self.log_path,
                        serialize_config(self.config),
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

    def _log_trial(self, trial: Trial=None, results: Dict[str, Any]=None, eval_results: Dataset=None, final_results: Dataset=None) -> Optional[int]:
        """Log trial results to database."""
        try:
            eval_id = int(time.time()*1000 + random.randint(1, 1000))
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN")
                try:
                    if self.module_type == "retriever":
                        self._log_retriever_trial(cursor, eval_id, trial, results)
                    elif self.module_type == "data_ingest":
                        self._log_data_ingest_trial(cursor, eval_id, trial, results)
                    elif self.module_type == "generation":
                        self._log_generation_trial(cursor, eval_id, trial, results)
                    
                    conn.commit()
                    return eval_id
                
                except Exception as e:
                    cursor.execute("ROLLBACK")
                    raise e
                
        except Exception as e:
            logger.error(f"Failed to log trial results: {e}")
            return None

    def _log_retriever_trial(self, cursor, eval_id: int, trial: Trial, results: Dict[str, Any]):
        """Log retriever trial results."""
        cursor.execute(
            """
            INSERT INTO retriever_eval_summary (
                eval_id, run_id, trial_number, config,
                avg_score, avg_context_precision, avg_context_recall,
                avg_latency, total_questions, successful_questions,
                error_rate, eval_start_ts, eval_end_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                eval_id, self.run_id, trial.number,
                json.dumps(results['config']),
                results['avg_score'],
                results['metrics']['avg_context_precision'],
                results['metrics']['avg_context_recall'],
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
            INSERT INTO retriever_eval_details (
                eval_id, question_id, question, contexts,
                ground_truth, context_precision, context_recall,
                f1_score, latency, error, eval_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    eval_id, idx,
                    detail['question'],
                    json.dumps(detail.get('contexts', [])),
                    detail.get('ground_truth', ''),
                    detail['metrics'].get('context_precision'),
                    detail['metrics'].get('context_recall'),
                    detail['metrics'].get('f1_score'),
                    detail.get('latency'),
                    detail.get('error'),
                    detail['eval_timestamp']
                )
                for idx, detail in enumerate(results['question_details'])
            ]
        )

    def _log_data_ingest_trial(self, cursor, eval_id: int, trial: Trial, results: Dict[str, Any]):
        """Log data ingest trial results."""
        cursor.execute(
            """
            INSERT INTO data_ingest_eval_summary (
                eval_id, run_id, trial_number, config,
                avg_score, avg_latency, total_questions,
                successful_questions, error_rate,
                eval_start_ts, eval_end_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                eval_id, self.run_id, trial.number,
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
        
        cursor.executemany(
            """
            INSERT INTO data_ingest_eval_details (
                eval_id, question_id, question, retrieved_chunks,
                relevance_scores, weighted_score, latency,
                error, eval_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    eval_id, idx,
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

    def _log_generation_trial(self, cursor, eval_id: int, trial: Trial, results: Dict[str, Any]):
        """Log generation trial results."""
        try:
            # Extract prompt_key and prompt based on summary type
            summary = results.get('summary', {})
            prompt_key = summary.get('prompt_key', '')
            prompt = summary.get('prompt', '')
            
            # Log summary metrics
            cursor.execute(
                """
                INSERT INTO generation_eval_summary (
                    run_id,
                    eval_id,
                    prompt_key,
                    prompt,
                    config,
                    average_correctness
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    eval_id,
                    prompt_key,
                    prompt,
                    json.dumps(results.get('config', {})),
                    results.get('score', 0.0)
                )
            )
            
            # Log detailed results if available
            detailed_results = results.get('detailed_results', [])
            if detailed_results:
                cursor.executemany(
                    """
                    INSERT INTO generation_eval_details (
                        eval_id,
                        question_id, 
                        question,
                        answer, 
                        ground_truth, 
                        prompt_key,
                        prompt, 
                        answer_correctness
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            eval_id,
                            idx,
                            detail.get('user_input', ''),
                            detail.get('response', ''),
                            detail.get('reference', ''),
                            prompt_key,
                            prompt,
                            detail.get('answer_correctness', 0.0)
                        )
                        for idx, detail in enumerate(detailed_results)
                    ]
                )
            
            logger.debug(f"Logged generation trial results for eval_id {eval_id}, trial {trial.number if trial else 'unknown'}")
        
        except Exception as e:
            logger.error(f"Failed to log generation trial results: {e}")
            raise

    def __call__(self, study: Study, trial: Trial, eval_results=None, final_results=None):
        """Called after each trial completion."""
        if study is not None and trial is not None:
            # Standard Optuna trial
            results = study.user_attrs.get(f"trial_{trial.number}_results", {})
            # logger.info(f"Results: {results}\n Type: {type(results)}")
            if results:
                self._log_trial(trial=trial, results=results)
                logger.debug(f"Logged results for trial {trial.number}")
        elif eval_results is not None or final_results is not None:
            # Legacy support for non-Optuna generation callbacks
            # Convert final_results to expected format if available
            if hasattr(final_results, 'model_dump'):
                results = {
                    'score': final_results.best_score,
                    'config': final_results.best_config.model_dump() if hasattr(final_results.best_config, 'model_dump') else {},
                    'summary': {
                        'prompt_key': getattr(final_results.best_config, 'prompt_key', ''),
                        'prompt': final_results.best_prompt
                    }
                }
                self._log_trial(results=results)
                logger.debug("Logged results for generation evaluation")
            else:
                logger.warning("No valid results format for generation logging")
        else:
            logger.warning("No valid study or trial data for logging")

    def __del__(self):
        """Update run status when optimization completes."""
        self._update_run_status("Success")