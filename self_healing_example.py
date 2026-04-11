import json
import os

from dotenv import load_dotenv

from self_healing_pipeline import SelfHealingPipeline


load_dotenv()


if __name__ == "__main__":
    pipeline = SelfHealingPipeline(neon_db_url=os.getenv("NEON_DATABASE_URL", "").strip() or None)
    try:
        result = pipeline.run("What is the latest news about OpenAI?")
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    finally:
        pipeline.close()
