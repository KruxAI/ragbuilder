from fastapi import FastAPI, Depends, HTTPException, Request, Path as PathParam
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.logger import logger 
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import sqlite3
import markdown
import threading
import webbrowser
import ssl
import time
import json
import os
import hashlib
import logging
import requests
import uvicorn
import warnings
import optuna
from pathlib import Path
from urllib.parse import urlparse
from ragbuilder.executor import rag_builder, rag_builder_bayes_optimization_optuna, rag_manager, get_model_obj
from ragbuilder.langchain_module.loader import loader as l
from ragbuilder.langchain_module.common import setup_logging, progress_state
from ragbuilder import generate_data
from ragbuilder.rag_templates.top_n_templates import get_templates
from ragbuilder.analytics import track_event
from ragbuilder.sampler import DataSampler
from ragbuilder.data_processor import DataProcessor
from ragbuilder.evaldb_dmls import *

# fastapi_setup_logging(logger)
setup_logging()
logger = logging.getLogger("ragbuilder")
LOG_FILENAME = logger.handlers[0].baseFilename
LOG_DIRNAME = Path(LOG_FILENAME).parent
BASE_DIR = Path(__file__).resolve().parent
logger.info(f"LOG_FILENAME = {LOG_FILENAME}")
url = "http://localhost:8005"

app = FastAPI()
DATABASE = 'eval.db'
BAYES_OPT = 1
CURRENT_RUN_ID = 0

templates = Jinja2Templates(directory=Path(BASE_DIR, 'templates'))
app.mount("/static", StaticFiles(directory=Path(BASE_DIR, 'static')), name="static")

warnings.filterwarnings(action="ignore", message=r"datetime.datetime.utcnow")

def basename(path):
    return os.path.basename(path)

# Register the filter with the Jinja2 environment
templates.env.filters['basename'] = basename

def get_db():
    try:
        db = sqlite3.connect(DATABASE, check_same_thread=False)
        db.row_factory = sqlite3.Row
        return db
    except sqlite3.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def get_hashmap(db: sqlite3.Connection = Depends(get_db)):
    cur = db.execute('SELECT hash, test_data_path FROM synthetic_data_hashmap')
    rows = cur.fetchall()
    return {row[0]: row[1] for row in rows}

def insert_hashmap(hash: str, path: str, db: sqlite3.Connection):
    logger.info(f"Saving hashmap for synthetic data: {path} ...")
    insert_query = """
        INSERT INTO synthetic_data_hashmap (hash, test_data_path) 
        VALUES (?, ?)
    """
    db.execute(insert_query, (hash, path))
    db.commit()
    logger.info(f"Saved hashmap for synthetic data: {path}")

def ensure_module_type_column():
    """Add module_type column if it doesn't exist."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        
        # Check if column exists
        cursor.execute("PRAGMA table_info(run_details)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'module_type' not in columns:
            try:
                # Add column with default value
                cursor.execute("""
                    ALTER TABLE run_details 
                    ADD COLUMN module_type VARCHAR(100) 
                    DEFAULT 'ui_workflow'
                """)
                conn.commit()
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise

@app.on_event("startup")
async def startup():
    db = get_db()
    db.execute(run_details_dml)
    db.execute(rag_eval_details_dml)
    db.execute(rag_eval_summary_dml)
    db.execute(synthetic_data_hashmap_dml)
    ensure_module_type_column()
    db.close()

# @app.on_event("shutdown")
# def shutdown_db():
#     db.close()

class ChatSessionCreate(BaseModel):
    eval_id: int

class ChatMessage(BaseModel):
    message: str

@app.post("/create_chat_session")
async def create_chat_session(session_data: ChatSessionCreate, db: sqlite3.Connection = Depends(get_db)):
    try:
        rag = rag_manager.get_rag(session_data.eval_id, db)
        session_id = f"{session_data.eval_id}_{int(time.time())}"
        return {"session_id": session_id}
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create chat session")
    
@app.get("/chat/{eval_id}", response_class=HTMLResponse)
async def chat_page(request: Request, eval_id: int):
    return templates.TemplateResponse(request=request, name="chat.html", context={"eval_id": eval_id})
    
@app.post("/chat/{session_id}")
async def chat(session_id: str, chat_message: ChatMessage, db: sqlite3.Connection = Depends(get_db)):
    try:
        eval_id = int(session_id.split('_')[0])
        rag = rag_manager.get_rag(eval_id, db)
        response = rag.invoke(chat_message.message)
        return {"answer": response["answer"]}
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail="Failed to process chat message")

@app.get("/", response_class=HTMLResponse)
def index(request: Request, db: sqlite3.Connection = Depends(get_db)):
    cur = db.execute("""
        SELECT 
            run_id,
            module_type,
            status,
            description,
            src_data,
            log_path,
            run_config,
            disabled_opts,
            datetime(run_ts, 'unixepoch', 'localtime') AS run_ts
        FROM run_details
        ORDER BY run_ts DESC
    """)
    runs = cur.fetchall()
    db.close()
    return templates.TemplateResponse(request=request, name='index.html', context={"runs": runs})
    # return templates.TemplateResponse(str(Path(BASE_DIR, 'templates', 'index.html')), {"request": request, "runs": runs})

@app.get('/summary/{run_id}', response_class=HTMLResponse)
async def summary(request: Request, run_id: int, db: sqlite3.Connection = Depends(get_db)):
    cur = db.execute("""
        SELECT 
            rag_eval_summary.run_id,
            run_details.description,
            rag_eval_summary.eval_id,
            rag_eval_summary.rag_config,
            rag_eval_summary.code_snippet,
            round(rag_eval_summary.avg_answer_correctness, 2) AS avg_answer_correctness,
            round(rag_eval_summary.avg_faithfulness, 2) AS avg_faithfulness,
            round(rag_eval_summary.avg_answer_relevancy, 2) AS avg_answer_relevancy,
            round(rag_eval_summary.avg_context_precision, 2) AS avg_context_precision,
            round(rag_eval_summary.avg_context_recall, 2) AS avg_context_recall,
            round(rag_eval_summary.avg_tokens, 0) AS avg_tokens,
            round(rag_eval_summary.avg_cost_per_query, 5) AS avg_cost_per_query,
            round(rag_eval_summary.avg_latency/1000000000.0, 2) AS avg_latency,
            datetime(rag_eval_summary.eval_ts, 'unixepoch', 'localtime') AS eval_ts
        FROM rag_eval_summary
        LEFT JOIN run_details 
            ON rag_eval_summary.run_id = run_details.run_id
        WHERE rag_eval_summary.run_id = ?
        ORDER BY 6 DESC
    """, (run_id,))
    evals = cur.fetchall()
    db.close()
    description = evals[0]['description'] if evals else "Unnamed Project"
    return templates.TemplateResponse(
        request=request, 
        name='summary.html', 
        context={
            "evals": evals, 
            "description": description,
            "run_id": run_id
        }
    )

@app.get('/details/{eval_id}', response_class=HTMLResponse)
async def details(request: Request, eval_id: int, db: sqlite3.Connection = Depends(get_db)):
    module_type = 'ui_workflow'
    cur = db.execute("""
        SELECT 
            question_id,
            question,
            answer,
            contexts,
            ground_truth,
            round(answer_correctness, 2) AS answer_correctness,
            round(faithfulness, 2) AS faithfulness,
            round(answer_relevancy, 2) AS answer_relevancy,
            round(context_precision, 2) AS context_precision,
            round(context_recall, 2) AS context_recall,
            round(latency/1000000000.0, 2) AS latency,
            round(tokens, 1) AS tokens,
            round(cost, 5) AS cost,
            datetime(eval_ts, 'unixepoch', 'localtime') as eval_timestamp
        FROM rag_eval_details 
        WHERE eval_id = ?
    """, (eval_id,))
    details = cur.fetchall()
    db.close()
    return templates.TemplateResponse(
        request=request, 
        name='details.html', 
        context={"details": details, "module_type": module_type}
    )

@app.get("/sdk/summary/{run_id}")
async def sdk_summary(request: Request, run_id: int, db: sqlite3.Connection = Depends(get_db)):
    """Show summary dashboard for SDK-based runs."""
    cursor = db.cursor()
        
    # Get run details
    cursor.execute("""
        SELECT module_type, description, src_data, run_config 
        FROM run_details 
        WHERE run_id = ?
    """, (run_id,))
    run_info = cursor.fetchone()
    
    if not run_info:
        raise HTTPException(status_code=404, detail="Run not found")
        
    module_type, description, src_data, run_config = run_info
    
    # Get evaluations based on module type
    if module_type == 'data_ingest':
        cursor.execute("""
            SELECT * FROM data_ingest_eval_summary 
            WHERE run_id = ? 
            ORDER BY avg_score desc
        """, (run_id,))
    elif module_type == 'retriever' :  # retriever
        cursor.execute("""
            SELECT * FROM retriever_eval_summary 
            WHERE run_id = ? 
            ORDER BY avg_score desc
        """, (run_id,))
    else:
        cursor.execute("""
            SELECT * FROM generation_eval_summary 
            WHERE run_id = ? 
            ORDER BY average_correctness desc
        """, (run_id,))

    evals = cursor.fetchall()
    test_var=[dict(zip([col[0] for col in cursor.description], row)) 
                     for row in evals]
    print("results:",test_var)
    db.close()

    try:
        if isinstance(run_config, str):
            run_config = json.loads(run_config)
        formatted_config = json.dumps(run_config, indent=2)
    except json.JSONDecodeError:
        formatted_config = run_config  # Fallback to original if parsing fails
    
    return templates.TemplateResponse(
        "sdk_summary.html",
        {
            "request": request,
            "run_id": run_id,
            "module_type": module_type,
            "description": description,
            "src_data": src_data,
            "run_config": formatted_config,
            "evals": [dict(zip([col[0] for col in cursor.description], row)) 
                     for row in evals]
        }
    )

@app.get("/sdk/details/{module_type}/{eval_id}")
async def sdk_details(
    request: Request, 
    eval_id: int, 
    module_type: str = PathParam(..., regex="^(data_ingest|retriever|generation)$"),
    db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()
    # Get details based on module type
    if module_type == 'data_ingest':
        cursor.execute("""
            SELECT 
                question_id,
                question,
                retrieved_chunks,
                relevance_scores,
                weighted_score,
                latency,
                error,
                datetime(eval_ts/1000.0, 'unixepoch', 'localtime') as eval_timestamp
            FROM data_ingest_eval_details
            WHERE eval_id = ?
            ORDER BY question_id
        """, (eval_id,))
    elif module_type == 'retriever' :
        cursor.execute("""
            SELECT 
                question_id,
                question,
                contexts,
                ground_truth,
                context_precision,
                context_recall,
                f1_score,
                latency,
                error,
                datetime(eval_ts/1000.0, 'unixepoch', 'localtime') as eval_timestamp
            FROM retriever_eval_details
            WHERE eval_id = ?
            ORDER BY question_id
        """, (eval_id,))
    else:
        cursor.execute("""
            SELECT 
                question_id,
                question,
                answer,
                ground_truth,
                prompt_key,
                prompt,
                answer_correctness
            FROM generation_eval_details
            WHERE eval_id = ?
            ORDER BY question_id
        """, (eval_id,))
#TODO datetime(eval_ts/1000.0, 'unixepoch', 'localtime') as eval_timestamp
    details = cursor.fetchall()
    db.close()
    
    return templates.TemplateResponse(
        "details.html",
        {
            "request": request,
            "module_type": module_type,
            "details": details
        }
    )

class TrialData(BaseModel):
    number: int
    values: List[Optional[float]]
    params: Dict[str, Any]
    state: str

class StudyData(BaseModel):
    study_name: str
    directions: List[str]
    pareto_front: List[TrialData]
    trials: List[TrialData]
    n_trials: int
    parameter_importance: Dict[str, List[float]]

@app.get("/api/study/{run_id}", response_model=StudyData)
def get_study_data(run_id: int, db: sqlite3.Connection = Depends(get_db)):
    logger.debug(f"Fetching module_type and description for run_id: {run_id}")
    try:
        cursor = db.cursor()
        
        # Get module_type and description (study_name)
        cursor.execute("""
            SELECT module_type, description 
            FROM run_details 
            WHERE run_id = ?
        """, (run_id,))
        result = cursor.fetchone()

        if not result:
            raise HTTPException(status_code=404, detail="Run not found")
        
        module_type, description = result

        study_name = (
            str(run_id) if module_type == 'legacy' 
            else description
        )
    except Exception as e:
        logger.error(f"Error fetching module_type and description from DB: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch module_type and description from DB")

    try:
        logger.debug(f"Loading study data for run_id: {run_id}")
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{DATABASE}")
        # print(f"Study data loaded: {study}")
        
        logger.debug(f"Setting up trials data...")
        trials_data = [
            TrialData(
                number=trial.number,
                values=trial.values if trial.values is not None else [],
                params=trial.params,
                state=trial.state.name
            )
            for trial in study.trials
        ]
        # print(f"Trials data set up: {trials_data}")

        logger.debug(f"Setting up pareto front data...")
        pareto_front = [
            TrialData(
                number=trial.number,
                values=trial.values,
                params=trial.params,
                state=trial.state.name
            )
            for trial in study.best_trials
        ]
        # print(f"Pareto front data set up: {pareto_front}")

        # Calculate parameter importance
        logger.debug(f"Calculating parameter importance...")
        importance = {}
        for i in range(len(study.directions)):
            importance_i = optuna.importance.get_param_importances(study, target=lambda t: t.values[i])
            for param, imp in importance_i.items():
                if param not in importance:
                    importance[param] = [0] * len(study.directions)
                importance[param][i] = imp
        # print(f"Parameter importance calculated: {importance}")
        
        logger.debug(f"Setting up study data...")
        # print(f"Study name: {study.study_name}, type = {type(study.study_name)}")
        # print(f"Directions: {[d.name for d in study.directions]}")
        # print(f"Pareto front: {pareto_front}")
        # # print(f"Trials data: {trials_data}")
        # print(f"Number of trials: {len(study.trials)}")
        # print(f"Parameter importance: {importance}")

        study_data = StudyData(
            study_name=study.study_name,
            directions=[d.name for d in study.directions],
            pareto_front=pareto_front,
            trials=trials_data,
            n_trials=len(study.trials),
            parameter_importance=importance
        )
        
        return study_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read_docs", response_class=RedirectResponse)
def docs():
    return RedirectResponse(url="https://github.com/KruxAI/ragbuilder/blob/main/README.md")

@app.get('/view_log/{filename}', response_class=HTMLResponse)
async def view_log(request: Request, filename: str):
    log_filepath = os.path.join(LOG_DIRNAME, filename)
    logger.info(f"Accessing log file at: {log_filepath}")
    try:
        with open(log_filepath, 'r') as file:
            log_content = file.read()
        return templates.TemplateResponse(request=request, name='log_view.html', context={"log_content": log_content, "filename": log_filepath})
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Log file not found")

@app.get("/get_log_filename")
async def get_log_filename():
    return {"log_filename": LOG_FILENAME}

@app.get("/get_log_updates")
def get_log_updates():
    with open(LOG_FILENAME, 'r') as log_file:
        log_content = log_file.read()
    return {"log_content": log_content}

@app.get("/progress")
def get_progress():
    return progress_state.get_progress()

@app.get("/get_current_run_id")
def get_current_run_id():
    if CURRENT_RUN_ID:
        return {"run_id": CURRENT_RUN_ID}
    else:
        raise HTTPException(status_code=404, detail="No active run ID found")

class SourceDataCheck(BaseModel):
    sourceData: str
    useSampling: Optional[bool] = Field(None)

def get_hash(source_data, use_sampling=False):
    src_type=l.classify_path(source_data)
    prefix = "sampled_" if use_sampling else ""
    
    if src_type == "directory":
        return _get_hash_dir(source_data, prefix)
    elif src_type == "url":
        return _get_hash_url(source_data, prefix)
    elif src_type == "file":
        return _get_hash_file(source_data, prefix)
    else:
        logger.error("Invalid input path type {source_data}")
        return None

def _get_hash_file(source_data, prefix=""):
    md5_hash = hashlib.md5(prefix.encode())
    with open(source_data, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def _get_hash_url(url, prefix=""):
    try:
        # Fetch the content of the URL
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers, allow_redirects=True)
        response.raise_for_status()  # Check for HTTP errors
        md5_hash = hashlib.md5(prefix.encode())
        md5_hash.update(response.content)
        return md5_hash.hexdigest()
    except requests.RequestException as e:
        logger.error(f"Error fetching URL: {e}")
        return None
    
def _get_hash_dir(dir_path, prefix=""):
    if not os.path.isdir(dir_path):
        logger.error(f"{dir_path} is not a valid directory.")
        return None
    md5_hash = hashlib.md5(prefix.encode())
    try:
        for root, dirs, files in os.walk(dir_path):
            for file_name in sorted(files):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'rb') as file:
                    while chunk := file.read(8192):
                        md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error hashing directory content: {e}")
        return None
    
@app.post("/check_test_data")
async def check_test_data(data: SourceDataCheck, hashmap: dict = Depends(get_hashmap)):
    source_data = data.sourceData
    use_sampling = data.useSampling
    hash_value = get_hash(source_data, use_sampling)
    if hash_value in hashmap:
        return {"exists": True, "path": hashmap[hash_value]}
    else:
        return {"exists": False, "hash": hash_value}

def _is_valid_source_data(source_data):
    # Check if source_data is a URL
    try:
        result = urlparse(source_data)
        if all([result.scheme, result.netloc]):
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'}
            response = requests.head(source_data, headers=headers, allow_redirects=True)
            return response.status_code == 200
    except:
        pass
    
    # Check if source_data is a file or directory
    expanded_path = os.path.expanduser(source_data)
    return os.path.exists(expanded_path)

@app.post("/check_source_data")
async def check_source_data(data: SourceDataCheck):
    try:
        is_valid_source_data = _is_valid_source_data(data.sourceData)
        logger.info(f"Source data {data.sourceData} validity: {is_valid_source_data}")
        if is_valid_source_data:
            logger.info(f"Estimating source data size...")
            sampler = DataSampler(data.sourceData)
            size = sampler.estimate_data_size()
            exceeds_threshold = sampler.need_sampling()
            logger.info(f"Source data size: {size}, exceeds_threshold: {exceeds_threshold}")
            return {
                "valid": is_valid_source_data,
                "size": size,
                "exceeds_threshold": exceeds_threshold
            }
        else:
            return {"valid": is_valid_source_data}
    except Exception as e:
        logger.error(f"Error checking source data: {e}")
        return {"valid": False}
    
@app.get("/templates")
def get_rag_templates():
    templates = get_templates()
    return JSONResponse(content={"templates": templates})

class ProjectData(BaseModel):
    description: str
    sourceData: str
    useSampling: bool
    compareTemplates: bool
    includeNonTemplated: bool
    selectedTemplates: List[str] = Field(default_factory=list)
    chunkingStrategy: dict[str, bool]
    chunkSize: dict[str, int]
    embeddingModel: dict[str, bool]
    huggingfaceEmbeddingModel: str
    azureOAIEmbeddingModel: str
    googleVertexAIEmbeddingModel: str
    ollamaEmbeddingModel: str
    vectorDB: str
    retriever: dict[str, bool]
    topK: dict[str, bool]
    llm: dict[str, bool]
    huggingfaceLLMModel: str
    groqLLMModel: str
    azureOAILLMModel: str
    googleVertexAILLMModel: str
    ollamaLLMModel: str
    generateSyntheticData: bool
    optimization: str
    evalFramework: str
    evalEmbedding: str
    evalLLM: str
    sotaEmbeddingModel: Optional[str] = Field(default=None)
    sotaLLMModel: Optional[str] = Field(default=None)
    compressors: Optional[dict[str, bool]] = Field(default=None)
    syntheticDataGeneration: Optional[dict] = Field(default=None)
    testDataPath: Optional[str] = Field(default=None)
    existingSynthDataPath: Optional[str] = Field(default=None)
    testSize: Optional[str] = Field(default=None)
    criticLLM: Optional[str] = Field(default=None)
    generatorLLM: Optional[str] = Field(default=None)
    generatorEmbedding: Optional[str] = Field(default=None)
    numRuns: Optional[str] = Field(default=None)
    nJobs: Optional[int] = Field(default=None)
    dataProcessors: Optional[List[str]] = Field(default=None)

@app.post("/rbuilder")
def rbuilder_route(project_data: ProjectData, db: sqlite3.Connection = Depends(get_db)):
    result = parse_config(project_data.model_dump(), db)
    return result
    # print(project_data)
    # print(project_data.model_dump())
    # return {"status": "success", "message": "Ok"}

def _get_disabled_opts(config: dict):
    disabled_opts = []
    stack = [config]  # Initialize the stack with the root dictionary
    while stack:
        current_dict = stack.pop()
        for key, value in current_dict.items():
            if isinstance(value, dict):
                stack.append(value)
            elif value is False:
                disabled_opts.append(key)
    return disabled_opts

def _db_write(run_details: tuple, db: sqlite3.Connection):
    logger.info(f"Saving run config details in db...")
    insert_query=f"""
            INSERT INTO run_details(
                run_id,
                status,
                description,
                src_data,
                log_path,
                run_config,
                disabled_opts,
                run_ts
            ) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
    db.execute(insert_query, run_details)
    db.commit() 
    logger.info(f"Saved run config details in db")

def _update_status(run_id: int, status: int, db: sqlite3.Connection):
    logger.info(f"Updating status for run_id {run_id} as {status}...")
    upd_query=f"""
            UPDATE run_details set
                status = ?
            WHERE run_id = ?
        """
    db.execute(upd_query, (status, run_id,))
    db.commit() 
    logger.info(f"Updated run_id {run_id} with status {status} in db")


def parse_config(config: dict, db: sqlite3.Connection):
    # The implementation remains largely the same, but use the passed db connection
    # Instead of get_db(), use the db parameter
    # Replace jsonify with direct dictionary returns
    # ...
    global CURRENT_RUN_ID
    enable_analytics = os.getenv('ENABLE_ANALYTICS', 'True').lower() == 'true'
    logger.info(f"enable_analytics = {enable_analytics}")
    if enable_analytics:
        track_event('0')
    logger.info(f"Initiating parsing config: {config}")
    disabled_opts=_get_disabled_opts(config)
    logger.info(f"Disabled options: {disabled_opts}")
    CURRENT_RUN_ID=int(time.time())
    desc=config["description"]
    compare_templates=config["compareTemplates"]
    include_granular_combos=config["includeNonTemplated"]
    selected_templates = config.get('selectedTemplates', [])
    # gen_synthetic_data=config["generateSyntheticData"]
    syntheticDataGenerationOpts=config.get("syntheticDataGeneration", None)
    existingSynthDataPath=config.get("existingSynthDataPath", None)
    vectorDB=config.get("vectorDB", None)
    hf_embedding=config.get("huggingfaceEmbeddingModel", None)
    azureoai_embedding=config.get("azureOAIEmbeddingModel", None)
    googlevertexai_embedding=config.get("googleVertexAIEmbeddingModel", None)
    ollama_embedding=config.get("ollamaEmbeddingModel", None)
    hf_llm=config.get("huggingfaceLLMModel", None)
    groq_llm=config.get("groqLLMModel", None)
    azureoai_llm=config.get("azureOAILLMModel", None)
    googlevertexai_llm=config.get("googleVertexAILLMModel", None)
    ollama_llm=config.get("ollamaLLMModel", None)
    min_chunk_size=int(config["chunkSize"]["min"])
    max_chunk_size=int(config["chunkSize"]["max"])
    optimization=config.get("optimization", 'fullParameterSearch')
    eval_framework = config.get('evalFramework')
    eval_embedding = config.get('evalEmbedding')
    eval_llm = config.get('evalLLM')
    other_embedding = [emb for emb in [hf_embedding, azureoai_embedding, googlevertexai_embedding, ollama_embedding] if emb is not None and emb != ""]
    other_llm = [llm for llm in [hf_llm, groq_llm, azureoai_llm, googlevertexai_llm, ollama_llm] if llm is not None and llm != ""]
    sota_embedding = config.get('sotaEmbeddingModel')
    sota_llm = config.get('sotaLLMModel')
    src_full_path = config.get("sourceData", None)
    use_sampling = config.get("useSampling", False)
    data_processors = config.get("dataProcessors", None)
    # build this array data_processors
    # Call DataSampler to sample data
    data_sampler = DataSampler(os.path.expanduser(src_full_path), enable_sampling=use_sampling)
    #Sample data and return sample path or orginal path
    src_path = data_sampler.sample_data()
    if data_processors is not None and len(data_processors) > 0:
        # Call DataProcessor to process data
        data_processor=DataProcessor(src_path, data_processors)
        # Process data and return process file path or orginal path
        src_path=data_processor.processed_data
    else:
        logger.info(f"No data processors selected. Using original data.")
        
    src_data={'source':'url','input_path': src_path}
    
    if existingSynthDataPath:
        f_name=existingSynthDataPath
        logger.info(f"Existing synthetic test data detected: {f_name}")
    elif syntheticDataGenerationOpts:
        try:
            test_size=int(config["syntheticDataGeneration"]["testSize"])
        except ValueError:
            logger.error(f'Expected integer value for test_size. Got {config["syntheticDataGeneration"]["testSize"]}')
            raise
        
        critic_llm=get_model_obj('llm', config["syntheticDataGeneration"]["criticLLM"], temperature = 0.2)
        generator_llm=get_model_obj('llm', config["syntheticDataGeneration"]["generatorLLM"], temperature = 0.2)
        embedding_model=get_model_obj('embedding', config["syntheticDataGeneration"]["generatorEmbedding"])
        # TODO: Add Distribution
        try:
            progress_state.toggle_synth_data_gen_progress(1)
            f_name=generate_data.generate_data(
                src_data=src_path,
                generator_model=generator_llm,
                critic_model=critic_llm,
                embedding_model=embedding_model,
                test_size=test_size
            )
            if f_name is None:
                logger.error(f'Synthetic test data generation failed.')
                # _update_status(run_id, 'Failed')
                return {
                    "status": "error",
                    # "message": str(e)
                }, 400
        except Exception as e:
            logger.error(f'Synthetic test data generation failed: {e}')
            raise
        # Insert generated into hashmap 
        insert_hashmap(get_hash(src_full_path, use_sampling), f_name, db)
        progress_state.toggle_synth_data_gen_progress(0)
        if enable_analytics:
            track_event('2')
        logger.info(f"Synthetic test data generation completed: {f_name}") 
    else:
        f_name=config["testDataPath"]
        logger.info(f"User provided test data: {f_name}")

    run_details=(
        CURRENT_RUN_ID, 
        'Running',
        desc, 
        src_path,
        LOG_FILENAME,
        json.dumps(config), 
        json.dumps(disabled_opts), 
        CURRENT_RUN_ID
    )
    _db_write(run_details, db)
    # return json.dumps({'status':'success', 'f_name': f_name})
    try:
        if optimization=='bayesianOptimization':
            logger.info(f"Using Bayesian optimization to find optimal RAG configs...")
            num_runs = int(config.get("numRuns", 50))
            n_jobs = int(config.get("nJobs", 1))
            res = rag_builder_bayes_optimization_optuna(
                run_id=CURRENT_RUN_ID, 
                compare_templates=compare_templates, 
                src_data=src_data, 
                test_data=f_name,
                selected_templates=selected_templates,
                include_granular_combos=include_granular_combos, 
                vectorDB=vectorDB,
                min_chunk_size=min_chunk_size, 
                max_chunk_size=max_chunk_size, 
                other_embedding=other_embedding,
                other_llm=other_llm,
                num_runs=num_runs,
                n_jobs=n_jobs,
                eval_framework=eval_framework,
                eval_embedding=eval_embedding,
                eval_llm=eval_llm,
                sota_embedding=sota_embedding,
                sota_llm=sota_llm,
                disabled_opts=disabled_opts
            )
        elif optimization=='fullParameterSearch' :
            logger.info(f"Spawning RAG configs by invoking rag_builder...")
            res = rag_builder(
                run_id=CURRENT_RUN_ID, 
                compare_templates=compare_templates, 
                src_data=src_data, 
                test_data=f_name,
                selected_templates=selected_templates,
                include_granular_combos=include_granular_combos, 
                vectorDB=vectorDB,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                other_embedding=other_embedding,
                other_llm=other_llm,
                eval_framework=eval_framework,
                eval_embedding=eval_embedding,
                eval_llm=eval_llm,
                sota_embedding=sota_embedding,
                sota_llm=sota_llm,
                disabled_opts=disabled_opts
            )
            logger.info(f"res = {res}")
        else:
            logger.error(f"Unknown optimization value.")
            return {
            "status": "error",
            "message": "Unknown optimization value"
        }, 400
    except Exception as e:
        logger.error(f'Failed to complete creation and evaluation of RAG configs: {e}')
        _update_status(CURRENT_RUN_ID, 'Failed', db)
        db.close()
        return {
            "status": "error",
            "message": str(e)
        }, 400
    else:
        logger.info(f"Processing finished successfully")
        _update_status(CURRENT_RUN_ID, 'Success', db)
        db.close()
        if enable_analytics:
            track_event('1')
        return {
            "status": "success",
            "message": "Completed successfully.",
            "run_id": CURRENT_RUN_ID
        }
    # return jsonify({'status':'success', 'f_name': f_name})

# def main():
#     threading.Timer(1.25, lambda: webbrowser.open(url)).start()
#     uvicorn.run(app, host="0.0.0.0", port=8005)

# if __name__ == '__main__':
#     main()

# Function to open URL without SSL verification

def is_docker():
    """Check if the code is running inside a Docker container."""
    path = '/.dockerenv'
    return os.path.exists(path)

def open_url(url):
    import urllib.request
    context = ssl._create_unverified_context()
    try:
        urllib.request.urlopen(url, context=context)
    except Exception as e:
        logger.error(f"Error opening URL: {e}")



def main():
    if is_docker():
        url = "http://0.0.0.0:55003"
        logging.info("Running inside Docker container")
        logging.info("Open http://0.0.0.0:55003 in your browser. Please open with appropriate port number if you have mapped another port.")
        threading.Timer(1.25, lambda: webbrowser.open(url)).start()
        uvicorn.run(app, host="0.0.0.0", port=8005)
    else:
        url = "http://127.0.0.1:8005"
        logging.info("Opening URL in browser")
        threading.Timer(1.25, lambda: webbrowser.open(url)).start()
        uvicorn.run(app, host="0.0.0.0", port=8005)

if __name__ == '__main__':
    main()