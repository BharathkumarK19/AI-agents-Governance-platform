import asyncio
import inspect


def resolve_agent_result(result):
    if not inspect.isawaitable(result):
        return result

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(result)

    raise RuntimeError(
        "Received an async CrewAI result while already inside a running event loop. "
        "Run the pipeline in a worker thread or await the result from async code."
    )
