from sethu.config import settings


def has_key() -> bool:
    key = settings.groq_api_key if settings.llm_provider == "groq" else settings.anthropic_api_key
    return bool(key)


def complete(system: str, user: str, max_tokens: int, low_effort: bool = False) -> str:
    """One-shot completion through the configured provider.

    low_effort asks reasoning models to think briefly; use it for small mechanical tasks.
    """
    if settings.llm_provider == "groq":
        from groq import Groq

        extra = {}
        if low_effort and settings.llm_model.startswith("openai/gpt-oss"):
            extra["reasoning_effort"] = "low"
        resp = Groq(api_key=settings.groq_api_key).chat.completions.create(
            model=settings.llm_model,
            max_completion_tokens=max_tokens,  # reasoning models spend part of this thinking
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **extra,
        )
        return resp.choices[0].message.content or ""

    import anthropic

    msg = anthropic.Anthropic(api_key=settings.anthropic_api_key).messages.create(
        model=settings.llm_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")
