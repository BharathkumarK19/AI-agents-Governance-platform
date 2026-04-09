import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agents.crewai_runtime import get_crewai_components


load_dotenv()

DEFAULT_MODEL = "openrouter/auto"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _require_env(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is not set.")
    return value


def get_llm():
    return ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=_require_env("OPENROUTER_API_KEY"),
        model=DEFAULT_MODEL,
        temperature=0.2,
    )


def get_crewai_llm():
    _, _, LLM, _, _ = get_crewai_components()
    return LLM(
        model=DEFAULT_MODEL,
        base_url=OPENROUTER_BASE_URL,
        api_key=_require_env("OPENROUTER_API_KEY"),
        temperature=0.2,
    )
