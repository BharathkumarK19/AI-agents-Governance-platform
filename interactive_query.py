from typing import Any


def ask_followup_questions(user_query, data_profile=None):
    prompts = _build_prompts(user_query, data_profile=data_profile or {})
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


def _build_prompts(user_query: str, data_profile: dict | None = None) -> list[tuple[str, str]]:
    normalized = user_query.lower()
    data_profile = data_profile or {}
    top_items = [item.get("item_name", "") for item in data_profile.get("top_items", []) if item.get("item_name")]
    top_customers = [item.get("customer_id", "") for item in data_profile.get("top_customers", []) if item.get("customer_id")]
    sample_items = ", ".join(top_items[:3]) if top_items else "top-selling items"
    sample_customers = ", ".join(top_customers[:3]) if top_customers else "known customers"

    prompts: list[tuple[str, str]] = []

    if any(term in normalized for term in ("compare", "vs", "versus")):
        prompts.append(("comparison_target", f"Which items, customers, or periods should I compare? Example: {sample_items}"))

    if any(term in normalized for term in ("customer", "buyer", "segment")):
        prompts.append(("customer_filter", f"Do you want to focus on a specific customer or segment? Example: {sample_customers}"))
    elif any(term in normalized for term in ("item", "product", "sku", "purchase")):
        prompts.append(("item_filter", f"Do you want to focus on a specific item or SKU? Example: {sample_items}"))

    if "profit" in normalized or "margin" in normalized:
        prompts.append(("focus_area", "Should I emphasize profit, margin, or both?"))
    elif any(term in normalized for term in ("sale", "sales", "revenue", "price")):
        prompts.append(("focus_area", "Should I emphasize revenue, pricing, or discount impact?"))
    elif "gst" in normalized or "tax" in normalized:
        prompts.append(("focus_area", "Should I emphasize GST amounts, GST compliance, or tax impact?"))
    else:
        prompts.append(("focus_area", "Which business angle matters most here: sales, profit, margin, discount, GST, or customers?"))

    if any(term in normalized for term in ("trend", "over time", "monthly", "daily", "weekly", "change")):
        prompts.append(("time_granularity", "Do you want the trend grouped daily, weekly, or monthly?"))

    prompts.append(("time_range", "What time range should I consider in days?"))
    prompts.append(("output_style", "Do you want a short summary, recommendations, or both?"))
    return prompts


def _extract_days(value: str) -> int | None:
    digits = "".join(character for character in value if character.isdigit())
    if not digits:
        return None
    return int(digits)
