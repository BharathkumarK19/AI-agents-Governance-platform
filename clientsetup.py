import os
from pathlib import Path

from openai import AuthenticationError, OpenAI


MODEL_NAME = "openai/gpt-oss-20b:free"


def _load_env_file() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY is not set.")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)


if __name__ == "__main__":
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": "What is the capital of France? Reply in one short sentence.",
                }
            ],
        )
        print(response.choices[0].message.content)
    except AuthenticationError as exc:
        raise RuntimeError(
            "OpenRouter authentication failed. Check `OPENROUTER_API_KEY` in `.env` "
            "and replace it with a valid active key."
        ) from exc
