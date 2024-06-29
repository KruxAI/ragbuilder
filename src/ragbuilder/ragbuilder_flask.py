import sqlite3
import markdown
import pkg_resources
import threading
import webbrowser
import time
import json
import os
import hashlib
import logging
import requests
from flask import Flask, render_template, render_template_string, g, request, jsonify, abort
from pathlib import Path
from urllib.parse import urlparse
from ragbuilder.executor import rag_builder
from ragbuilder.langchain_module.loader import loader as l
from ragbuilder.langchain_module.common import setup_logging
from ragbuilder import generate_data
from ragbuilder.analytics import track_event
from ragbuilder.evaldb_dmls import *

setup_logging()
logger = logging.getLogger("ragbuilder")
LOG_FILENAME=logger.handlers[0].baseFilename
LOG_DIRNAME=Path(LOG_FILENAME).parent
print(f"LOG_FILENAME = {LOG_FILENAME}")

url = "http://localhost:8001"

app=Flask(__name__)
DATABASE = 'eval.db'

def basename(path):
    return os.path.basename(path)

# Register the filter with the Jinja2 environment
app.jinja_env.filters['basename'] = basename

def get_hashmap():
    hashmap = getattr(g, '_hashmap', None)
    if hashmap is None:
        db = get_db()
        cur=db.execute('SELECT hash, test_data_path FROM synthetic_data_hashmap')
        rows = cur.fetchall()
        hashmap = g._hashmap = {row[0]: row[1] for row in rows}
    return hashmap

def insert_hashmap(hash, path):
    logger.info(f"Saving hashmap for synthetic data: {path} ...")
    db = get_db()
    insert_query=f"""
            INSERT INTO synthetic_data_hashmap (hash, test_data_path) 
            VALUES ('{hash}', '{path}')
        """
    db.execute(insert_query)
    db.commit() 
    logger.info(f"Saved hashmap for synthetic data: {path}")


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route("/")
def index():
    db = get_db()
    tbl_create_run_details= db.execute(run_details_dml)
    tbl_create_rag_eval_details= db.execute(rag_eval_details_dml)
    tbl_create_rag_eval_summary= db.execute(rag_eval_summary_dml)
    tbl_create_synthetic_data_hashmap= db.execute(synthetic_data_hashmap_dml)
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
            ORDER BY run_ts DESC""")
    runs = cur.fetchall()
    return render_template('index.html', runs=runs)

@app.route('/summary/<int:run_id>')
def summary(run_id):
    db = get_db()
    cur = db.execute("""
            SELECT 
                rag_eval_summary.run_id,
                run_details.description,
                rag_eval_summary.eval_id,
                rag_eval_summary.rag_config,
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
            ORDER BY 5 DESC""", (run_id,))
    evals = cur.fetchall()
    description = evals[0]['description'] if evals else "Unnamed Project"
    return render_template('summary.html', evals=evals, description=description)


@app.route('/details/<int:eval_id>')
def details(eval_id):
    db = get_db()
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
            WHERE eval_id = ?""", (eval_id,))
    details = cur.fetchall()
    return render_template('details.html', details=details)

@app.route("/docs")
def docs():
    with open("README.md", "r") as f:
        markdown_text = f.read()
    html = markdown.markdown(markdown_text)    
    return render_template('docs.html', content=html)
    # return html

@app.route('/view_log/<path:filename>')
def view_log(filename):
    log_filepath = os.path.join(LOG_DIRNAME, filename)
    print(f"Accessing log file at: {log_filepath}")  # Debugging output
    try:
        with open(log_filepath, 'r') as file:
            log_content = file.read()
        return render_template('log_view.html', log_content=log_content, filename=log_filepath)
    except FileNotFoundError:
        abort(404)

@app.route("/get_log_filename", methods=["GET"])
def get_log_filename():
    return jsonify({"log_filename": LOG_FILENAME})

@app.route("/get_log_updates", methods=["GET"])
def get_log_updates():
    with open(LOG_FILENAME, 'r') as log_file:
        log_content = log_file.read()
    return jsonify({"log_content": log_content})

@app.route("/check_test_data", methods=["POST"])
def check_test_data():
    source_data = request.json.get('sourceData')
    hash_value = get_hash(source_data)
    hashmap=get_hashmap()
    if hash_value in hashmap:
        return jsonify({"exists": True, "path": hashmap[hash_value]})
    else:
        return jsonify({"exists": False, "hash": hash_value})

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
        print(f"Error hashing directory content: {e}")
        return None

@app.route("/check_source_data", methods=["POST"])
def check_source_data():
    source_data = request.json.get('sourceData')
    if is_valid_source_data(source_data):
        return jsonify({"valid": True})
    else:
        return jsonify({"valid": False})

def is_valid_source_data(source_data):
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

@app.route("/rbuilder", methods=["POST"])
def rbuilder_route():
    project_data = request.json
    result=parse_config(project_data)
    return result
    # Use below for debugging
    # dummy_test(project_data)
    # print(f'project_data={project_data}')
    # return jsonify({"status": "success"})

def _get_disabled_opts(config):
    disabled_opts = []
    stack = [config]  # Initialize the stack with the root dictionary
    while stack:
        current_dict = stack.pop()
        for key, value in current_dict.items():
            if isinstance(value, dict):
                stack.append(value)
            elif not value:
                disabled_opts.append(key)
    return disabled_opts

def _db_write(run_details):
    logger.info(f"Logging run config details in db...")
    db = get_db()
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

def _update_status(run_id, status):
    logger.info(f"Updating status for run_id {run_id} as {status}...")
    db = get_db()
    upd_query=f"""
            UPDATE run_details set
                status = ?
            WHERE run_id = ?
        """
    db.execute(upd_query, (status, run_id,))
    db.commit() 
    logger.info(f"Saved run config details in db")


def parse_config(config):
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

        # print(f'test_data_type={type(test_size)}\
        #       test_size=\"{test_size}\",\
        #     generator_llm=\"{generator_llm}\",\
        #     critic_llm=\"{critic_llm}\",\
        #     embeddings=\"{embedding_model}\"')
        # return jsonify({'status':'success'})
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
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 400
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
    _db_write(run_details)
    try:
        logger.info(f"Spawning RAG configs by invoking rag_builder...")
        res = rag_builder(
            run_id=run_id, 
            compare_templates=compare_templates, 
            src_data=src_data, 
            test_data=f_name,
            include_granular_combos=include_granular_combos, 
            vectorDB=vectorDB,
            disabled_opts=disabled_opts
        )
        logger.info(f"res = {res}")
    except Exception as e:
        logger.error(f'Failed to complete creation and evaluation of RAG configs: {e}')
        _update_status(run_id, 'Failed')
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
    else:
        logger.info(f"Processing finished successfully")
        _update_status(run_id, 'Success')
        return jsonify({
            "status": "success",
            "message": "Completed successfully.",
            "run_id": run_id
        })
    # return jsonify({'status':'success', 'f_name': f_name})

 
def main():
    # logger.info("Open http://localhost:8001/ in your browser to access the ragbuilder Dashboard.")
    threading.Timer(1.25, lambda: webbrowser.open(url)).start()
    app.run(port=8001)

if __name__ == '__main__':
    main()