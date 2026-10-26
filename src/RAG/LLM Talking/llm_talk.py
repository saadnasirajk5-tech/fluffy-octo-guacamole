
# =============================================================================
# CELL: Implement LLM providers (OpenAI + local)
# =============================================================================

class LLMProvider(ABC):
    """The interface every LLM provider must implement."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send a system+user prompt to the LLM and return its answer text."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The human-readable name of the model this provider talks to."""
        ...


class OpenAIProvider(LLMProvider):
    """Calls OpenAI's cloud chat completions API."""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=config.OPENAI_API_KEY or None)
        self._model = config.OPENAI_MODEL

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.LLM_TEMPERATURE,
        )
        return response.choices[0].message.content

    @property
    def model_name(self) -> str:
        return self._model


class LocalLLMProvider(LLMProvider):
    """Calls a local LLM server (Ollama, vLLM, llama.cpp, etc.).

    All these servers expose an OpenAI-compatible API at
    http://localhost:11434/v1/chat/completions (Ollama), so we talk to them
    with the same message format using plain HTTP.
    """

    def __init__(self):
        import httpx
        self.base_url = config.LOCAL_LLM_BASE_URL.rstrip("/")
        self._model = config.LOCAL_LLM_MODEL
        self.client = httpx.Client(base_url=self.base_url, timeout=120.0)
        self._headers = {"Authorization": "Bearer local", "Content-Type": "application/json"}

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": config.LLM_TEMPERATURE,
        }
        resp = self.client.post("/chat/completions", json=payload, headers=self._headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    @property
    def model_name(self) -> str:
        return self._model


def get_llm_provider(provider: str = None) -> LLMProvider:
    """Return the configured LLM provider."""
    name = (provider or config.LLM_PROVIDER).lower()
    if name == "openai":
        if not config.openai_configured:
            raise RuntimeError(
                "OpenAI is selected but OPENAI_API_KEY is not set. "
                "Set it in the environment or switch LLM_PROVIDER to 'local'."
            )
        return OpenAIProvider()
    if name == "local":
        return LocalLLMProvider()
    raise ValueError(f"Unknown LLM provider: {name!r}")


# --- Demo: show the provider we'd use (don't call the LLM without a key). ---
print(f"Configured provider: {config.LLM_PROVIDER}")
if config.LLM_PROVIDER == "openai" and not config.openai_configured:
    print("  (No OPENAI_API_KEY set - generation will fail gracefully.)")
    print("  Tip: set OPENAI_API_KEY in your environment, or switch to 'local'.")








































