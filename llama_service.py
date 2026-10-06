from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover - handled at runtime.
    Llama = None

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_NAME = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
MIN_MODEL_BYTES = 500 * 1024 * 1024
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME
DEFAULT_MODEL_URL = (
    "https://huggingface.co/bartowski/"
    "Llama-3.2-3B-Instruct-GGUF/resolve/main/"
    f"{DEFAULT_MODEL_NAME}?download=true"
)


@dataclass(slots=True)
class LlamaResult:
    ok: bool
    text: str = ""
    error: str | None = None
    mode: str = "local"
    raw: Any = None


def is_mock_mode() -> bool:
    return os.getenv("VISTA_DEV_MOCK_LLM", "0") == "1"


def get_model_path() -> Path:
    model_path = os.getenv("VISTA_LLM_MODEL_PATH")
    if model_path:
        return Path(model_path)
    model_name = os.getenv("VISTA_LLM_MODEL_NAME", DEFAULT_MODEL_NAME)
    return MODELS_DIR / model_name


if DEFAULT_MODEL_PATH.exists() and not os.getenv("VISTA_LLM_MODEL_PATH"):
    os.environ["VISTA_LLM_MODEL_PATH"] = str(DEFAULT_MODEL_PATH)


def get_model_url() -> str:
    return os.getenv("VISTA_LLM_MODEL_URL", DEFAULT_MODEL_URL)


def model_status() -> dict[str, str | bool]:
    model_path = get_model_path()

    if model_path.exists() and model_path.stat().st_size < MIN_MODEL_BYTES:
        return {
            "available": False,
            "mode": "incomplete",
            "message": (
                "Local Llama model file looks incomplete or still downloading. "
                "Wait for the GGUF file to finish downloading before enabling real inference."
            ),
            "model_path": str(model_path),
        }

    if is_mock_mode():
        return {
            "available": True,
            "mode": "mock",
            "message": "Development mock LLM mode is enabled.",
            "model_path": str(model_path),
        }

    if model_path.exists():
        return {
            "available": True,
            "mode": "local",
            "message": "Local GGUF model is available.",
            "model_path": str(model_path),
        }

    return {
        "available": False,
        "mode": "missing",
        "message": (
            "Local Llama model is not installed yet. "
            "Set VISTA_LLM_MODEL_PATH to the downloaded GGUF file."
        ),
        "model_path": str(model_path),
    }


class LocalLlamaService:
    def __init__(
        self,
        model_path: Path | None = None,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
    ) -> None:
        self.model_path = Path(model_path) if model_path else get_model_path()
        self.n_ctx = n_ctx
        self.n_threads = n_threads or max(1, os.cpu_count() or 1)
        self.n_gpu_layers = n_gpu_layers
        self._llm: Llama | None = None

    @property
    def available(self) -> bool:
        return self.model_path.exists() and self.model_path.stat().st_size >= MIN_MODEL_BYTES

    def load_model(self) -> Llama:
        if self._llm is not None:
            return self._llm

        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not installed in the active environment."
            )

        if not self.available:
            raise FileNotFoundError(
                f"Local Llama model is missing: {self.model_path}"
            )

        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            chat_format="llama-3",
            verbose=False,
        )
        return self._llm

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "You are a concise assistant for the VISTA app.",
        max_tokens: int = 128,
        temperature: float = 0.2,
    ) -> LlamaResult:
        if is_mock_mode():
            return LlamaResult(
                ok=True,
                text=f"[MOCK LLM] {user_prompt.strip()}",
                mode="mock",
            )

        if not self.available:
            return LlamaResult(
                ok=False,
                error=(
                    "Local Llama model is not installed yet. "
                    f"Expected GGUF file: {self.model_path}"
                ),
                mode="missing",
            )

        try:
            llm = self.load_model()
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # pragma: no cover - runtime path.
            return LlamaResult(ok=False, error=str(exc), mode="error")

        text = response["choices"][0]["message"]["content"].strip()
        return LlamaResult(ok=True, text=text, mode="local", raw=response)

    def generate_json(
        self,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> LlamaResult:
        return self.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )


_SERVICE_SINGLETON: LocalLlamaService | None = None


def get_local_llama_service() -> LocalLlamaService:
    global _SERVICE_SINGLETON
    if _SERVICE_SINGLETON is None:
        _SERVICE_SINGLETON = LocalLlamaService()
    return _SERVICE_SINGLETON


def format_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def test_local_llama(prompt: str = "Reply with exactly: local llama ready.") -> str:
    service = LocalLlamaService()
    result = service.generate(prompt, max_tokens=16, temperature=0.0)
    if not result.ok:
        raise RuntimeError(result.error or "Llama generation failed.")
    return result.text


if __name__ == "__main__":
    print(test_local_llama())