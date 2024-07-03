from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.logger import logger 
from pydantic import BaseModel, Field
from typing import Optional
import sqlite3
import markdown
import threading
import webbrowser
import time
import json
import os
import hashlib
import logging
import requests
import uvicorn
from pathlib import Path
from urllib.parse import urlparse
from ragbuilder.executor import rag_builder, rag_builder_bayes_optmization
from ragbuilder.langchain_module.loader import loader as l
from ragbuilder.langchain_module.common import setup_logging
from ragbuilder import generate_data
from ragbuilder.analytics import track_event
from ragbuilder.evaldb_dmls import *

# fastapi_setup_logging(logger)
setup_logging()
logger = logging.getLogger("ragbuilder")
LOG_FILENAME = logger.handlers[0].baseFilename
LOG_DIRNAME = Path(LOG_FILENAME).parent
BASE_DIR = Path(__file__).resolve().parent
print(f"LOG_FILENAME = {LOG_FILENAME}")

url = "http://localhost:8005"

app = FastAPI()
DATABASE = 'eval.db'
BAYES_OPT=1

templates = Jinja2Templates(directory=Path(BASE_DIR, 'templates'))
app.mount("/static", StaticFiles(directory=Path(BASE_DIR, 'static')), name="static")

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

def insert_hashmap(hash: str, path: str, db: sqlite3.Connection = Depends(get_db)):
    logger.info(f"Saving hashmap for synthetic data: {path} ...")
    insert_query = """
        INSERT INTO synthetic_data_hashmap (hash, test_data_path) 
        VALUES (?, ?)
    """
    db.execute(insert_query, (hash, path))
    db.commit()
    logger.info(f"Saved hashmap for synthetic data: {path}")

@app.on_event("startup")
async def startup():
    db = get_db()
    db.execute(run_details_dml)
    db.execute(rag_eval_details_dml)
    db.execute(rag_eval_summary_dml)
    db.execute(synthetic_data_hashmap_dml)
    db.close()

# @app.on_event("shutdown")
# def shutdown_db():
#     db.close()

@app.get("/", response_class=HTMLResponse)
def index(request: Request, db: sqlite3.Connection = Depends(get_db)):
    cur = db.execute("""
        SELECT 
            run_id,
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
    return templates.TemplateResponse(request=request, name='summary.html', context={"evals": evals, "description": description})

@app.get('/details/{eval_id}', response_class=HTMLResponse)
async def details(request: Request, eval_id: int, db: sqlite3.Connection = Depends(get_db)):
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
    return templates.TemplateResponse(request=request, name='details.html', context={"details": details})

@app.get("/read_docs", response_class=HTMLResponse)
def docs(request: Request):
    with open("README.md", "r") as f:
        markdown_text = f.read()
    html = markdown.markdown(markdown_text)
    return templates.TemplateResponse(request=request, name='docs.html', context={"content": html})

@app.get('/view_log/{filename}', response_class=HTMLResponse)
async def view_log(request: Request, filename: str):
    log_filepath = os.path.join(LOG_DIRNAME, filename)
    print(f"Accessing log file at: {log_filepath}")
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

class SourceDataCheck(BaseModel):
    sourceData: str

def get_hash(source_data):
    src_type=l.classify_path(source_data)
    
    if src_type == "directory":
        return _get_hash_dir(source_data)
    elif src_type == "url":
        return _get_hash_url(source_data)
    elif src_type == "file":
        return _get_hash_file(source_data)
    else:
        logger.error("Invalid input path type {source_data}")
        return None

def _get_hash_file(source_data):
    return hashlib.md5(open(source_data, "rb").read()).hexdigest()
    # return hashlib.md5(source_data.encode()).hexdigest()

def _get_hash_url(url):
    try:
        # Fetch the content of the URL
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        return hashlib.md5(response.content).hexdigest()
    
    except requests.RequestException as e:
        logger.error(f"Error fetching URL: {e}")
        return None
    
def _get_hash_dir(dir_path):
    if not os.path.isdir(dir_path):
        logger.error(f"{dir_path} is not a valid directory.")
        return None
    md5_hash = hashlib.md5()
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
    hash_value = get_hash(source_data)
    if hash_value in hashmap:
        return {"exists": True, "path": hashmap[hash_value]}
    else:
        return {"exists": False, "hash": hash_value}

def _is_valid_source_data(source_data):
    # Check if source_data is a URL
    try:
        result = urlparse(source_data)
        if all([result.scheme, result.netloc]):
            response = requests.head(source_data)
            return response.status_code == 200
    except:
        pass
    
    # Check if source_data is a file or directory
    return os.path.exists(source_data)

@app.post("/check_source_data")
async def check_source_data(data: SourceDataCheck):
    return {"valid": _is_valid_source_data(data.sourceData)}

class ProjectData(BaseModel):
    description: str
    sourceData: str
    compareTemplates: bool
    includeNonTemplated: bool
    chunkingStrategy: dict[str, bool]
    chunkSize: dict[str, str]
    embeddingModel: dict[str, bool]
    vectorDB: str
    retriever: dict[str, bool]
    topK: dict[str, bool]
    contextualCompression: bool
    llm: dict[str, bool]
    generateSyntheticData: bool
    optimization: str
    compressors: Optional[dict[str, bool]] = Field(default=None)
    syntheticDataGeneration: Optional[dict] = Field(default=None)
    testDataPath: Optional[str] = Field(default=None)
    existingSynthDataPath: Optional[str] = Field(default=None)
    testSize: Optional[str] = Field(default=None)
    criticLLM: Optional[str] = Field(default=None)
    generatorLLM: Optional[str] = Field(default=None)
    embedding: Optional[str] = Field(default=None)

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
    logger.info(f"Logging run config details in db...")
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
    enable_analytics = os.getenv('ENABLE_ANALYTICS', 'True').lower() == 'true'
    logger.info(f"enable_analytics= {enable_analytics}")
    if enable_analytics:
        track_event(config)
    logger.info(f"Initiating parsing config: {config}")
    disabled_opts=_get_disabled_opts(config)
    logger.info(f"Disabled options: {disabled_opts}")
    run_id=int(time.time())
    desc=config["description"]
    compare_templates=config["compareTemplates"]
    include_granular_combos=config["includeNonTemplated"]
    # gen_synthetic_data=config["generateSyntheticData"]
    src_path=config.get("sourceData", None)
    src_data={'source':'url','input_path': src_path}
    syntheticDataGenerationOpts=config.get("syntheticDataGeneration", None)
    existingSynthDataPath=config.get("existingSynthDataPath", None)
    vectorDB=config.get("vectorDB", None)
    min_chunk_size=int(config["chunkSize"]["min"])
    max_chunk_size=int(config["chunkSize"]["max"])
    optimization=config.get("optimization", 'fullParameterSearch')
    
    if existingSynthDataPath:
        f_name=existingSynthDataPath
        logger.info(f"Existing synthetic test data detected: {f_name}")
    elif syntheticDataGenerationOpts:
        try:
            test_size=int(config["syntheticDataGeneration"]["testSize"])
        except ValueError:
            logger.error(f'Expected integer value for test_size. Got {config["syntheticDataGeneration"]["testSize"]}')
            raise
        
        critic_llm=config["syntheticDataGeneration"]["criticLLM"]
        generator_llm=config["syntheticDataGeneration"]["generatorLLM"]
        embedding_model=config["syntheticDataGeneration"]["embedding"]
        # TODO: Add Distribution

        try:
            f_name=generate_data.generate_data(
                src_data=src_path,
                test_size=test_size,
                generator_model=generator_llm,
                critic_model=critic_llm,
                embedding_model=embedding_model
            )
            if f_name is None:
                logger.error(f'Synthetic test data generation failed.')
                # _update_status(run_id, 'Failed')
                return {
                    "status": "error",
                    "message": str(e)
                }, 400
        except Exception as e:
            logger.error(f'Synthetic test data generation failed: {e}')
            raise
        # Insert generated into hashmap 
        insert_hashmap(get_hash(src_path), f_name)
        g._hashmap=None # To refresh hashmap
        logger.info(f"Synthetic test data generation completed: {f_name}")

        
    else:
        f_name=config["testDataPath"]
        logger.info(f"User provided test data: {f_name}")

    run_details=(
        run_id, 
        'Running',
        desc, 
        src_path,
        LOG_FILENAME,
        json.dumps(config), 
        json.dumps(disabled_opts), 
        run_id
    )
    _db_write(run_details, db)

    try:
        if include_granular_combos and optimization=='bayesianOptimization':
            logger.info(f"Using Bayesian optimization to find optimal RAG configs...")
            res = rag_builder_bayes_optmization(
                run_id=run_id, 
                compare_templates=compare_templates, 
                src_data=src_data, 
                test_data=f_name,
                include_granular_combos=include_granular_combos, 
                vectorDB=vectorDB,
                min_chunk_size=min_chunk_size, 
                max_chunk_size=max_chunk_size, 
                disabled_opts=disabled_opts
            )
        if compare_templates or (include_granular_combos  and optimization=='fullParameterSearch') :
            logger.info(f"Spawning RAG configs by invoking rag_builder...")
            res = rag_builder(
                run_id=run_id, 
                compare_templates=compare_templates, 
                src_data=src_data, 
                test_data=f_name,
                include_granular_combos=include_granular_combos, 
                vectorDB=vectorDB,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                disabled_opts=disabled_opts
            )
            logger.info(f"res = {res}")
    except Exception as e:
        logger.error(f'Failed to complete creation and evaluation of RAG configs: {e}')
        _update_status(run_id, 'Failed', db)
        db.close()
        return {
            "status": "error",
            "message": str(e)
        }, 400
    else:
        logger.info(f"Processing finished successfully")
        _update_status(run_id, 'Success', db)
        db.close()
        return {
            "status": "success",
            "message": "Completed successfully.",
            "run_id": run_id
        }
    # return jsonify({'status':'success', 'f_name': f_name})

def main():
    threading.Timer(1.25, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=8005)

if __name__ == '__main__':
    main()