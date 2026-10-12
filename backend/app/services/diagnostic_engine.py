"""Diagnostic analyzer addressing real user issues with AirLLM."""
import re
from typing import List
from packaging import version
from app.models.schemas import DiagnosticRequest, DiagnosticResult, DiagnosticIssue

class DiagnosticEngine:
    @classmethod
    def analyze_environment(cls, req: DiagnosticRequest) -> DiagnosticResult:
        issues: List[DiagnosticIssue] = []

        # 1. Transformers version check: _is_stateful incompatibility in >= 4.40
        try:
            tf_v = version.parse(req.transformers_version)
            if tf_v >= version.parse("4.40.0"):
                issues.append(
                    DiagnosticIssue(
                        category="transformers_compatibility",
                        severity="error",
                        summary="AttributeError: 'AirLLMLlama2' has no attribute '_is_stateful'",
                        root_cause="Hugging Face Transformers >= 4.40.0 requires GenerationMixin subclasses to declare _is_stateful = False.",
                        recommended_fix="Set `_is_stateful = False` in AirLLMBaseModel definition, or pin transformers to ~4.39.0."
                    )
                )
            if tf_v >= version.parse("4.36.0"):
                issues.append(
                    DiagnosticIssue(
                        category="transformers_cache",
                        severity="warning",
                        summary="TypeError: 'DynamicCache' object is not subscriptable",
                        root_cause="Transformers >= 4.36 replaced legacy tuple kv_cache with DynamicCache objects incompatible with direct index slicing.",
                        recommended_fix="Convert DynamicCache to legacy tuple format or use cache.to_legacy_cache() before layer indexing."
                    )
                )
        except Exception:
            pass

        # 2. CUDA vs ROCm checks
        if req.device_type.lower() == "rocm":
            issues.append(
                DiagnosticIssue(
                    category="hardware_backend",
                    severity="warning",
                    summary="ROCm GPU detected: Potential slow fallback or missing Triton/bitsandbytes",
                    root_cause="BitsAndBytes 4-bit/8-bit compression on AMD ROCm requires specialized builds (e.g. bitsandbytes-rocm) and proper HSA_OVERRIDE_GFX_VERSION.",
                    recommended_fix="Install bitsandbytes for ROCm and verify GPU kernel dispatch with ROCM_PATH and HIP_VISIBLE_DEVICES."
                )
            )

        if not req.has_cuda and req.device_type.lower() == "cuda":
            issues.append(
                DiagnosticIssue(
                    category="cuda_configuration",
                    severity="error",
                    summary="Torch not compiled with CUDA enabled",
                    root_cause="PyTorch installed from PyPI default CPU wheel rather than CUDA-enabled index.",
                    recommended_fix="Reinstall torch with: `pip install torch --index-url https://download.pytorch.org/whl/cu121`"
                )
            )

        # 3. Log pattern parsing if raw error logs are provided
        if req.error_log:
            log = req.error_log
            if "rotary_emb" in log or "rotary_fn" in log:
                issues.append(
                    DiagnosticIssue(
                        category="modeling_attention",
                        severity="error",
                        summary="AttributeError: 'LlamaAttention' object has no attribute 'rotary_emb'",
                        root_cause="Recent Llama implementations relocated rotary embedding application into forward function or RotaryEmbedding module.",
                        recommended_fix="Update modeling layer hook to query `self.rotary_emb` conditionally or call rotary_fn."
                    )
                )
            if "BetterTransformer" in log or "BetterTrasnform" in log:
                issues.append(
                    DiagnosticIssue(
                        category="optimum_bettertransformer",
                        severity="warning",
                        summary="BetterTransformer import or NoneType invocation failure",
                        root_cause="Setting BetterTransformer = None causes runtime failures when callers invoke methods on it.",
                        recommended_fix="Use PyTorch 2.0 native SDPA (Scaled Dot-Product Attention) via `attn_implementation='sdpa'` instead of BetterTransformer."
                    )
                )
            if "sentencepiece" in log.lower():
                issues.append(
                    DiagnosticIssue(
                        category="missing_dependency",
                        severity="error",
                        summary="Missing 'sentencepiece' tokenizer dependency",
                        root_cause="Llama tokenizer requires sentencepiece binary for SentencePieceProcessor.",
                        recommended_fix="Run `pip install sentencepiece protobuf` in your environment."
                    )
                )

        has_critical = any(i.severity == "error" for i in issues)
        score = max(0.0, 100.0 - (len([i for i in issues if i.severity == 'error']) * 35.0) - (len([i for i in issues if i.severity == 'warning']) * 15.0))

        return DiagnosticResult(
            issues_found=issues,
            is_compatible=not has_critical,
            compatibility_score=round(score, 1)
        )
