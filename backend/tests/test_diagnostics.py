import pytest
from app.services.diagnostic_engine import DiagnosticEngine
from app.models.schemas import DiagnosticRequest

def test_transformers_stateful_mixin_issue():
    req = DiagnosticRequest(
        transformers_version="4.41.2",
        torch_version="2.3.0",
        has_cuda=True,
        device_type="cuda"
    )
    res = DiagnosticEngine.analyze_environment(req)
    # Should catch _is_stateful error
    summaries = [i.summary for i in res.issues_found]
    assert any("_is_stateful" in s for s in summaries)
    assert res.is_compatible is False

def test_dynamic_cache_warning():
    req = DiagnosticRequest(
        transformers_version="4.37.0",
        torch_version="2.2.0",
        has_cuda=True,
        device_type="cuda"
    )
    res = DiagnosticEngine.analyze_environment(req)
    summaries = [i.summary for i in res.issues_found]
    assert any("DynamicCache" in s for s in summaries)

def test_cuda_not_enabled():
    req = DiagnosticRequest(
        transformers_version="4.38.0",
        torch_version="2.2.0",
        has_cuda=False,
        device_type="cuda"
    )
    res = DiagnosticEngine.analyze_environment(req)
    summaries = [i.summary for i in res.issues_found]
    assert any("not compiled with CUDA" in s for s in summaries)

def test_error_log_parsing():
    raw_log = "AttributeError: 'LlamaAttention' object has no attribute 'rotary_emb'\nModuleNotFoundError: No module named 'sentencepiece'"
    req = DiagnosticRequest(
        transformers_version="4.35.0",
        torch_version="2.1.0",
        has_cuda=True,
        device_type="cuda",
        error_log=raw_log
    )
    res = DiagnosticEngine.analyze_environment(req)
    summaries = [i.summary for i in res.issues_found]
    assert any("rotary_emb" in s for s in summaries)
    assert any("sentencepiece" in s for s in summaries)
