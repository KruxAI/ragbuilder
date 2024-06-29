run_details_dml= """CREATE  TABLE IF NOT Exists
  run_details (
    run_id BIGINT,
    status varchar(30),
    description varchar(4000),
    src_data varchar(4000),
    log_path varchar(4000),
    run_config JSON,
    disabled_opts JSON,
    run_ts BIGINT
  )"""


rag_eval_details_dml= """CREATE TABLE IF NOT EXISTS
  rag_eval_details (
    question_id BIGINT,
    eval_id BIGINT,
    run_id BIGINT,
    -- rag_config JSON,
    eval_ts BIGINT,
    question TEXT,
    answer TEXT,
    contexts TEXT,
    ground_truth TEXT,
    answer_correctness DOUBLE,
    faithfulness DOUBLE,
    answer_relevancy DOUBLE,
    context_precision DOUBLE,
    context_recall DOUBLE,
    latency DOUBLE,
    tokens INTEGER,
    cost DOUBLE
  )"""


rag_eval_summary_dml="""CREATE TABLE IF NOT EXISTS
  rag_eval_summary (
    run_id BIGINT,
    eval_id BIGINT PRIMARY KEY,
    rag_config JSON,
    code_snippet TEXT,
    avg_answer_correctness DOUBLE,
    avg_faithfulness DOUBLE,
    avg_answer_relevancy DOUBLE,
    avg_context_precision DOUBLE,
    avg_context_recall DOUBLE,
    avg_tokens DOUBLE,
    avg_cost_per_query DOUBLE,
    avg_latency DOUBLE,
    eval_ts BIGINT
  )"""


synthetic_data_hashmap_dml="""CREATE TABLE IF NOT EXISTS
  synthetic_data_hashmap (hash varchar(32), test_data_path varchar(4000))"""