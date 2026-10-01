import base64
import socket
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from ragbuilder import network
from ragbuilder.security import LocalAccessMiddleware, validate_bind_address, validate_source_path
from ragbuilder.langchain_module.llms.llmConfig import getLLM
from ragbuilder.langchain_module.embedding_model.embedding import getEmbedding
from ragbuilder.langchain_module.loader.loader import ragbuilder_url_loader, ragbuilder_file_loader


def client(monkeypatch, token=None, host="127.0.0.1", peer="127.0.0.1"):
    monkeypatch.delenv("RAGBUILDER_API_TOKEN", raising=False)
    if token is not None:
        monkeypatch.setenv("RAGBUILDER_API_TOKEN", token)
    app = FastAPI()
    app.add_middleware(LocalAccessMiddleware)

    @app.get("/")
    def root():
        return {"ok": True}

    return TestClient(app, base_url=f"http://{host}", client=(peer, 12345))


def test_local_access_and_browser_boundaries(monkeypatch):
    c = client(monkeypatch)
    assert c.get("/").status_code == 200
    assert c.get("/", headers={"Origin": "https://attacker.example"}).status_code == 403
    assert c.get("/", headers={"Host": "attacker.example"}).status_code == 403
    assert client(monkeypatch, peer="192.0.2.3").get("/").status_code == 403


def test_token_required_on_all_routes(monkeypatch):
    token = "test-token-" * 4
    c = client(monkeypatch, token, host="service.example", peer="192.0.2.3")
    for path in ["/", "/docs", "/openapi.json"]:
        assert c.get(path).status_code == 401
    assert c.get("/", headers={"Authorization": "Bearer " + token}).status_code == 200
    basic = base64.b64encode(("ragbuilder:" + token).encode()).decode()
    assert c.get("/", headers={"Authorization": "Basic " + basic}).status_code == 200
    assert c.get("/", headers={"Authorization": "Basic !!!!"}).status_code == 401
    assert client(monkeypatch, "short").get("/").status_code == 503


def test_network_binding_requires_strong_token(monkeypatch):
    monkeypatch.delenv("RAGBUILDER_API_TOKEN", raising=False)
    validate_bind_address("127.0.0.1")
    with pytest.raises(ValueError):
        validate_bind_address("0.0.0.0")
    monkeypatch.setenv("RAGBUILDER_API_TOKEN", "x" * 32)
    validate_bind_address("0.0.0.0")


def test_file_root_traversal_hidden_files_and_symlinks(monkeypatch, tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("RAGBUILDER_DATA_ROOT", str(root))
    assert validate_source_path("docs.txt") == str(root / "docs.txt")
    for path in ["../secret", ".env", str(tmp_path / "outside")]:
        with pytest.raises(ValueError):
            validate_source_path(path)
    (root / "escape").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError):
        validate_source_path(str(root))


@pytest.mark.parametrize("value", ["' + str(marker.append('executed')) + '", "x'); marker.append('executed'); #", "a\\b'\"\nmodel"])
def test_model_names_are_data_not_code(value):
    marker = []
    for generator, target, constructor in [(getLLM, "llm", "ChatOpenAI"), (getEmbedding, "embedding", "OpenAIEmbeddings")]:
        kwargs = {"retrieval_model" if target == "llm" else "embedding_model": "OpenAI:" + value}
        generated = generator(**kwargs)
        namespace = {constructor: lambda **args: args, "marker": marker}
        exec(generated["code_string"], namespace)
        assert namespace[target]["model"] == value
    assert marker == []


def test_loader_paths_are_data_not_code():
    value = "https://example.com/'); marker.append('executed'); #"
    marker = []

    class Loader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return [self.path]

    for generator, name in [(ragbuilder_url_loader, "WebBaseLoader"), (ragbuilder_file_loader, "UnstructuredFileLoader")]:
        namespace = {name: Loader, "marker": marker}
        exec(generator(value)["code_string"], namespace)
        assert namespace["docs"] == [value]
    assert marker == []


def dns(monkeypatch, addresses):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443)) for ip in addresses])


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "::ffff:127.0.0.1", "224.0.0.1"])
def test_nonpublic_destinations_rejected(monkeypatch, address):
    dns(monkeypatch, [address])
    with pytest.raises(ValueError):
        network.public_destination("https://example.com/")


def test_mixed_public_private_dns_rejected(monkeypatch):
    dns(monkeypatch, ["1.1.1.1", "127.0.0.1"])
    with pytest.raises(ValueError):
        network.public_destination("https://example.com/")


def test_pinned_connection_and_redirect_revalidation(monkeypatch):
    dns(monkeypatch, ["1.1.1.1"])
    seen = []

    class Pool:
        def __init__(self, **kwargs):
            seen.append(kwargs)

        def urlopen(self, *args, **kwargs):
            assert kwargs["redirect"] is False
            assert kwargs["headers"]["Host"] == "example.com"
            dns(monkeypatch, ["127.0.0.1"])
            return type("Redirect", (), {"status": 302, "headers": {"Location": "https://internal.example/"}, "close": lambda self: None})()

        def close(self):
            pass

    monkeypatch.setattr(network.urllib3, "HTTPSConnectionPool", Pool)
    with pytest.raises(ValueError):
        network.public_get("https://example.com/")
    assert len(seen) == 1
    assert seen[0]["host"] == "1.1.1.1"
    assert seen[0]["assert_hostname"] == "example.com"
    assert seen[0]["server_hostname"] == "example.com"


def test_download_size_limit(monkeypatch):
    dns(monkeypatch, ["1.1.1.1"])
    monkeypatch.setattr(network, "MAX_DOWNLOAD_BYTES", 4)

    class Raw:
        status = 200
        headers = {}

        def stream(self, *args, **kwargs):
            yield b"12345"

        def close(self):
            pass

    class Pool:
        def __init__(self, **kwargs):
            pass

        def urlopen(self, *args, **kwargs):
            return Raw()

        def close(self):
            pass

    monkeypatch.setattr(network.urllib3, "HTTPSConnectionPool", Pool)
    with pytest.raises(ValueError, match="size or time"):
        network.public_get("https://example.com")
