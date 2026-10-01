import importlib
import json

import pytest

usage = importlib.import_module("ragbuilder.core.telemetry")


def test_opt_out_has_no_network_or_identity_side_effects(monkeypatch):
    monkeypatch.setenv("ENABLE_ANALYTICS", "false")
    monkeypatch.setenv("RAGBUILDER_TELEMETRY_URL", "https://collector.example/events")
    monkeypatch.setattr(usage.RAGBuilderTelemetry, "_get_or_create_user_id", lambda self: pytest.fail("created identity"))
    telemetry = usage.RAGBuilderTelemetry()
    with telemetry.optimization_span("ragbuilder", {}):
        pass
    assert not telemetry.enabled
    assert telemetry._worker is None


def test_default_enabled_and_no_sensitive_fields(monkeypatch, tmp_path):
    monkeypatch.delenv("ENABLE_ANALYTICS", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGBUILDER_TELEMETRY_URL", raising=False)
    monkeypatch.setattr(usage, "user_data_dir", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(usage.RAGBuilderTelemetry, "_drain", lambda self: None)
    telemetry = usage.RAGBuilderTelemetry()
    assert telemetry.enabled
    assert telemetry.endpoint == usage.DEFAULT_TELEMETRY_ENDPOINT
    secret = "sensitive-user-content"
    with pytest.raises(RuntimeError):
        with telemetry.optimization_span("ragbuilder", {"api_key": secret}) as span:
            span.set_attribute("error_message", secret)
            raise RuntimeError(secret)
    telemetry.track_error("ragbuilder", RuntimeError(secret), {"source": secret})
    telemetry.eval_datagen_span(generator_model=secret)
    records = []
    while not telemetry._queue.empty():
        records.append(telemetry._queue.get_nowait())
        telemetry._queue.task_done()
    assert records[0]["event"] == "installation_started"
    assert secret not in json.dumps(records)
    assert all(set(record) == {"event", "module", "installation_id", "version"} for record in records)
    telemetry.shutdown()


def test_invalid_endpoint_disables_network(monkeypatch):
    monkeypatch.setenv("ENABLE_ANALYTICS", "true")
    for endpoint in ["", "http://collector.example/events", "https://user:secret@collector.example/events", "https://[invalid"]:
        monkeypatch.setenv("RAGBUILDER_TELEMETRY_URL", endpoint)
        assert not usage.RAGBuilderTelemetry().enabled


def test_dotenv_opt_out_is_honored(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ENABLE_ANALYTICS", raising=False)
    (tmp_path / ".env").write_text("ENABLE_ANALYTICS=false\n")
    monkeypatch.setenv("RAGBUILDER_TELEMETRY_URL", "https://collector.example/events")
    assert not usage.RAGBuilderTelemetry().enabled
