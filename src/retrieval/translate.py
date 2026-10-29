import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
_llm = None
_cache = {}


def to_german(query):
    """Translate a question into German legal wording. Returns None on failure."""
    global _llm
    if query in _cache:
        return _cache[query]
    try:
        if _llm is None:
            _llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model_name="openai/gpt-oss-20b",
                temperature=0,
            )
        prompt = (
            "Translate this question into German, using the terms a German "
            "commercial-law text (the HGB) would use. "
            "Reply with the German question only.\n\n" + query
        )
        text = _llm.invoke(prompt).content.strip()
        _cache[query] = text or None
    except Exception:
        _cache[query] = None
    return _cache[query]