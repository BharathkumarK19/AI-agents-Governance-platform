import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agents.crewai_runtime import get_crewai_components


load_dotenv()

DEFAULT_MODEL = "openrouter/auto"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MAX_TOKENS = 1024


def _require_env(name):
    value = os.getenv(name)
    if value is not None:
        value = value.strip()
    if not value:
        raise ValueError(f"{name} is not set.")
    return value


def _max_tokens() -> int:
    raw = os.getenv("OPENROUTER_MAX_TOKENS", "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return DEFAULT_MAX_TOKENS


def get_llm(model: str | None = None):
    return ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=_require_env("OPENROUTER_API_KEY"),
        model=(model or DEFAULT_MODEL),
        temperature=0.2,
        max_tokens=_max_tokens(),
    )


def get_crewai_llm(model: str | None = None):
    _, _, LLM, _, _ = get_crewai_components()
    return LLM(
        model=(model or DEFAULT_MODEL),
        base_url=OPENROUTER_BASE_URL,
        api_key=_require_env("OPENROUTER_API_KEY"),
        temperature=0.2,
        max_tokens=_max_tokens(),
    )
