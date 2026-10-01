from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from starlette.testclient import TestClient


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
