"""Advisory service supporting OpenAI-compatible, Anthropic, Gemini, and Ollama."""
import json
from typing import Dict, Any, Optional
import httpx
from app.core.config import settings
from app.models.schemas import AdvisoryRequest, AdvisoryResponse

class LLMAdvisoryService:
    @staticmethod
    def generate_offline_advisory(context_type: str, data: Dict[str, Any]) -> str:
        """Deterministic offline rule-based advisory when no LLM key is supplied."""
        if context_type == "profile":
            model = data.get("model_name", "LLM")
            fits = data.get("fits_in_target_vram", False)
            single_mb = data.get("single_layer_vram_mb", 0)
            full_gb = data.get("full_model_vram_gb", 0)
            headroom = data.get("vram_headroom_mb", 0)
            status_text = "fits comfortably" if fits else "exceeds VRAM buffer"
            return (
                f"[Deterministic Analysis] Profiling for {model}: The full model would require ~{full_gb} GB VRAM. "
                f"With AirLLM sequential layer loading, peak single-layer consumption is {single_mb} MB, which "
                f"{status_text} in the configured target GPU memory (headroom: {headroom} MB). "
                f"Recommendation: Enable multi-layer batching to saturate available GPU memory."
            )
        elif context_type == "schedule":
            optimal = data.get("optimal_layers_per_batch", 1)
            util = data.get("vram_utilization_pct", 0)
            speedup = data.get("speedup_vs_single_layer", 1.0)
            return (
                f"[Deterministic Analysis] Schedule optimization recommends batching {optimal} layers simultaneously. "
                f"This reaches {util}% VRAM utilization and provides a theoretical {speedup}x PCIe transfer speedup "
                f"over naive single-layer swapping by amortizing kernel launch latency."
            )
        elif context_type == "diagnostic":
            score = data.get("compatibility_score", 100)
            issues_cnt = len(data.get("issues_found", []))
            return (
                f"[Deterministic Analysis] Environment compatibility score is {score}/100. "
                f"Detected {issues_cnt} potential runtime issues. Review recommended fixes for Transformers mixins "
                f"and CUDA/ROCm execution runtimes to prevent runtime generation crashes."
            )
        return "[Deterministic Analysis] Offline telemetry analysis complete."

    @classmethod
    async def get_advisory(cls, req: AdvisoryRequest) -> AdvisoryResponse:
        # If no key is set, use deterministic offline advisory
        if not settings.LLM_API_KEY:
            offline_text = cls.generate_offline_advisory(req.context_type, req.context_data)
            return AdvisoryResponse(
                advisory_text=offline_text,
                provider=settings.LLM_PROVIDER,
                model=settings.LLM_MODEL,
                is_simulated=True,
                grounded_record_id=str(req.context_data.get("id", "none"))
            )

        # Real outbound LLM request based on provider
        prompt_content = (
            f"Analyze this AirLLM inference state:\n"
            f"Type: {req.context_type}\n"
            f"Data: {json.dumps(req.context_data)}\n"
            f"User notes: {req.user_prompt or 'None'}\n"
            f"Provide a concise, grounded technical recommendation."
        )

        headers = {}
        payload = {}
        url = settings.LLM_BASE_URL.rstrip("/")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if settings.LLM_PROVIDER == "anthropic":
                    url = f"{url}/v1/messages"
                    headers = {
                        "x-api-key": settings.LLM_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    }
                    payload = {
                        "model": settings.LLM_MODEL,
                        "max_tokens": 512,
                        "messages": [{"role": "user", "content": prompt_content}]
                    }
                    resp = await client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    text = data["content"][0]["text"]
                elif settings.LLM_PROVIDER == "gemini":
                    url = f"{url}/v1beta/models/{settings.LLM_MODEL}:generateContent?key={settings.LLM_API_KEY}"
                    payload = {
                        "contents": [{"parts": [{"text": prompt_content}]}]
                    }
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    # Default: openai-compatible or ollama
                    url = f"{url}/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {settings.LLM_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": settings.LLM_MODEL,
                        "messages": [
                            {"role": "system", "content": "You are an expert AI infrastructure engineer advising on low-VRAM AirLLM execution."},
                            {"role": "user", "content": prompt_content}
                        ],
                        "max_tokens": 512
                    }
                    resp = await client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"]

                return AdvisoryResponse(
                    advisory_text=text,
                    provider=settings.LLM_PROVIDER,
                    model=settings.LLM_MODEL,
                    is_simulated=False,
                    grounded_record_id=str(req.context_data.get("id", "none"))
                )
        except Exception as e:
            # Never crash or fabricate a successful response on provider error; redact credentials
            return AdvisoryResponse(
                advisory_text=f"Provider call failed ({settings.LLM_PROVIDER}): {type(e).__name__}. Falling back to offline advice:\n" +
                              cls.generate_offline_advisory(req.context_type, req.context_data),
                provider=settings.LLM_PROVIDER,
                model=settings.LLM_MODEL,
                is_simulated=True,
                grounded_record_id=str(req.context_data.get("id", "none"))
            )
