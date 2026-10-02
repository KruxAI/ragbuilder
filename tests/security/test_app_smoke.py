from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from starlette.testclient import TestClient


def test_package_contains_ui_and_prompt_resources():
    from importlib.resources import files
    package = files("ragbuilder")
    for path in ["templates/index.html", "templates/chat.html", "static/main.js", "generation/prompts.yaml"]:
        assert package.joinpath(path).read_bytes()


def test_sdk_generation_with_current_langchain():
    from ragbuilder import RAGBuilder
    from ragbuilder.generation.pipeline import GenerationPipeline

    assert RAGBuilder
    config = SimpleNamespace(llm=SimpleNamespace(llm=FakeListChatModel(responses=["test answer"])), prompt_template="Use this context: {context}")
    retriever = RunnableLambda(lambda query: [Document(page_content="test context")])
    pipeline = GenerationPipeline(config, retriever)
    result = pipeline.batch_generate(["test question"])[0]
    assert result == {"answer": "test answer", "context": "test context"}


def test_ui_routes_apply_security_and_source_validation(monkeypatch, tmp_path):
    from ragbuilder import ragbuilder as ui

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAGBUILDER_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("RAGBUILDER_API_TOKEN", raising=False)
    with TestClient(ui.app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/get_log_updates", headers={"Host": "attacker.example"}).status_code == 403
        assert client.post("/check_source_data", json={"sourceData": "../secret"}).status_code == 422
        assert client.post("/check_test_data", json={"sourceData": ".env"}).status_code == 422
        monkeypatch.setenv("RAGBUILDER_API_TOKEN", "x" * 32)
        for route in ["/", "/get_log_updates", "/get_log_filename", "/openapi.json"]:
            assert client.get(route).status_code == 401
        assert client.get("/", headers={"Authorization": "Bearer " + "x" * 32}).status_code == 200


def test_legacy_exec_keeps_imports_in_function_namespace():
    from ragbuilder.executor import _exec

    code = "import math\ndef rag_pipeline():\n    return math.sqrt(9)\n"
    assert _exec(code) == 3


def test_generated_vector_and_model_code_compiles():
    from ragbuilder.langchain_module.vectordb.vectordb import getVectorDB
    from ragbuilder.langchain_module.llms.llmConfig import getLLM

    for provider in ["OpenAI", "AzureOAI", "Google", "GoogleVertexAI", "Groq", "Mistral", "HF", "Ollama"]:
        code = getLLM(retrieval_model=provider + ":test-model")
        compile(code["import_string"] + "\n" + code["code_string"], provider, "exec")
    for database in ["chromaDB", "faissDB", "milvusDB", "qdrantDB", "weaviateDB", "singleStoreDB", "pineconeDB", "pgvector"]:
        code = getVectorDB(database, "text-embedding-3-small")
        compile(code["import_string"] + "\n" + code["code_string"], database, "exec")


def test_sota_substitution_does_not_expand_placeholders_in_user_data(monkeypatch):
    from ragbuilder.langchain_module.rag import getCode

    model = "{embedding_class}"
    source = "{llm_class}"
    monkeypatch.setattr(getCode, "ragbuilder_loader", lambda **kwargs: {
        "import_string": "", "code_string": "docs = " + repr(kwargs["input_path"]),
    })
    code = getCode.sota_code_mod(
        code="def rag_pipeline():\n        {loader_class}\n        {llm_class}\n        {embedding_class}\n        return docs, llm, embedding",
        input_path=source, retrieval_model="OpenAI:" + model,
        embedding_kwargs={"embedding_model": "OpenAI:test"},
    )
    namespace = {}
    exec(code, namespace)
    namespace["ChatOpenAI"] = lambda **kwargs: kwargs["model"]
    namespace["OpenAIEmbeddings"] = lambda **kwargs: kwargs["model"]
    assert namespace["rag_pipeline"]() == (source, model, "test")


def test_legacy_evaluation_preserves_database_fields_and_averages_scores(monkeypatch):
    import inspect
    import math
    from datasets import Dataset
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import EvaluationResult
    from ragas.callbacks import ChainRun
    from ragbuilder import eval as legacy

    rows = [{"question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "a",
             "eval_id": 1, "run_id": 2, "eval_ts": 3, "latency": 4, "tokens": 5, "cost": 6}]
    current_result = EvaluationResult(
        dataset=EvaluationDataset.from_list([{"user_input": "q", "response": "a", "retrieved_contexts": ["c"], "reference": "a"}]),
        scores=[{"answer_correctness": 0.75}],
        ragas_traces={"test": ChainRun(run_id="test", parent_run_id=None, name="evaluation", inputs={}, metadata={})},
    )
    def evaluate_without_model_calls(*args, **kwargs):
        inspect.signature(evaluate).bind(*args, **kwargs)
        return current_result
    monkeypatch.setattr(legacy, "evaluate", evaluate_without_model_calls)
    evaluator = legacy.RagEvaluator.__new__(legacy.RagEvaluator)
    evaluator.eval_dataset = Dataset.from_list(rows)
    evaluator.id = 1
    evaluator.llm = evaluator.embeddings = evaluator.run_config = None
    evaluator.prepare_eval_dataset = lambda: evaluator.eval_dataset
    evaluator._db_write = lambda: None
    assert evaluator.evaluate() is current_result
    expected = {**rows[0], "contexts": "c", "answer_correctness": 0.75}
    assert evaluator.result_df.iloc[0].to_dict() == expected
    assert legacy.answer_correctness_score(current_result) == 0.75
    assert math.isnan(legacy.answer_correctness_score(SimpleNamespace(scores=[{"answer_correctness": None}])))
    partial = SimpleNamespace(scores=[{"answer_correctness": n} for n in [1, 0.5, 1, 0.5, None]])
    assert legacy.answer_correctness_score(partial) == 0.75


def test_hybrid_template_runs_with_local_test_components(monkeypatch):
    from langchain_classic import hub
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.retrievers import BaseRetriever
    import langchain_chroma
    from ragbuilder.rag_templates.sota.hybrid_rag import code

    docs = [Document(page_content="test document with context")]
    class Retriever(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager):
            return docs
    class VectorStore:
        @classmethod
        def from_documents(cls, **kwargs):
            return cls()
        def as_retriever(self, **kwargs):
            return Retriever()
    monkeypatch.setattr(langchain_chroma, "Chroma", VectorStore)
    monkeypatch.setattr(hub, "pull", lambda name: ChatPromptTemplate.from_template("{context}\n{question}"))
    code = code.replace("{llm_class}", "llm = test_llm").replace("{loader_class}", "docs = test_docs").replace("{embedding_class}", "embedding = None")
    namespace = {"test_llm": FakeListChatModel(responses=["test answer"]), "test_docs": docs}
    exec(code, namespace)
    pipeline = namespace["rag_pipeline"]()
    assert pipeline.invoke("test")["answer"] == "test answer"
