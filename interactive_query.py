from typing import Any


def ask_followup_questions(user_query):
    prompts = _build_prompts(user_query)
    answers = {}

    for key, question in prompts:
        answers[key] = input(f"{question} ").strip()

    days = _extract_days(answers.get("time_range", ""))
    if days is not None:
        answers["days"] = days

    return answers


def generate_insight(metrics, user_answers, llm):
    prompt = (
        "Analyze business performance:\n\n"
        f"Metrics: {metrics}\n"
        f"User Context: {user_answers}\n\n"
        "Explain:\n"
        "1. What is happening\n"
        "2. What is going wrong\n"
        "3. What should be done next"
    )

    if hasattr(llm, "invoke"):
        response = llm.invoke(prompt)
        return getattr(response, "content", str(response))
    if hasattr(llm, "predict"):
        return llm.predict(prompt)
    if callable(llm):
        response = llm(prompt)
        return getattr(response, "content", str(response))

    raise TypeError("The provided LLM does not support invoke, predict, or direct calls.")


def _build_prompts(user_query: str) -> list[tuple[str, str]]:
    normalized = user_query.lower()

    if "profit" in normalized:
        return [
            ("focus_area", "Do you want gross profit, net profit, or margin analysis?"),
            ("time_range", "What time range should I consider in days?"),
            ("output_style", "Do you want recommendations, insights, or both?"),
        ]

    if "sale" in normalized or "revenue" in normalized:
        return [
            ("focus_area", "Should I focus on revenue trends, conversion efficiency, or discount impact?"),
            ("time_range", "What time range should I consider in days?"),
            ("output_style", "Do you want recommendations, insights, or both?"),
        ]

    return [
        ("focus_area", "Do you want analysis based on profit, sales, or efficiency?"),
        ("time_range", "What time range should I consider in days?"),
        ("output_style", "Do you want recommendations, insights, or both?"),
    ]


def _extract_days(value: str) -> int | None:
    digits = "".join(character for character in value if character.isdigit())
    if not digits:
        return None
    return int(digits)
