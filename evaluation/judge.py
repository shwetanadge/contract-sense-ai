import json
import os
import re
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

JUDGE_MODEL = "openai/gpt-oss-120b"
judge_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name=JUDGE_MODEL,
    temperature=0,
)


def _ask_json(prompt, retries=3):
    """Ask the judge and parse its JSON reply. Retries only temporary problems."""
    for attempt in range(retries):
        try:
            text = judge_llm.invoke(prompt).content
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group(0))
        except Exception as e:
            message = str(e)
            if "404" in message or "does not exist" in message:
                raise RuntimeError(f"Judge model '{JUDGE_MODEL}' is not available: {message[:150]}")
            print(f"   judge retry {attempt + 1}: {message[:80]}")
            time.sleep(8 * (attempt + 1))
    return None


def faithfulness(answer, context):
    """Share of the answer's claims that the context supports (0 to 1)."""
    prompt = f"""You are a strict fact checker.
List every factual claim in the ANSWER. For each claim, say whether it is
supported by the CONTEXT alone (true/false). Ignore citation markers like [1].

CONTEXT:
{context}

ANSWER:
{answer}

Reply with JSON only, in this format:
{{"claims": [{{"claim": "...", "supported": true}}]}}"""
    result = _ask_json(prompt)
    if not result or not result.get("claims"):
        return None
    claims = result["claims"]
    return sum(1 for c in claims if c.get("supported")) / len(claims)


def context_recall(expected, context):
    """Share of the expected answer's facts found in the retrieved context (0 to 1)."""
    prompt = f"""Split the EXPECTED ANSWER into its key factual statements.
For each statement, say whether it can be found in the CONTEXT (true/false).

CONTEXT:
{context}

EXPECTED ANSWER:
{expected}

Reply with JSON only, in this format:
{{"statements": [{{"statement": "...", "found": true}}]}}"""
    result = _ask_json(prompt)
    if not result or not result.get("statements"):
        return None
    items = result["statements"]
    return sum(1 for s in items if s.get("found")) / len(items)


def correctness(answer, expected):
    """How well the answer matches the expected answer: 0, 0.5 or 1."""
    prompt = f"""Compare the ANSWER with the EXPECTED ANSWER.
Score 1 if the answer contains all key points of the expected answer correctly.
Score 0.5 if it is correct but misses an important point, or has a minor error.
Score 0 if it is wrong, contradicts the expected answer, or refuses to answer.
Extra correct detail is fine and must not be penalised.

EXPECTED ANSWER:
{expected}

ANSWER:
{answer}

Reply with JSON only, in this format:
{{"score": 1, "reason": "one short sentence"}}"""
    result = _ask_json(prompt)
    if not result or "score" not in result:
        return None
    return float(result["score"])


if __name__ == "__main__":
    context = "(a) the processing is carried out by a public authority or body; (b) core activities require regular and systematic monitoring of data subjects on a large scale."
    answer = "A DPO is required for public authorities and for large-scale monitoring. It is also required for every company with more than 50 employees."
    expected = "A DPO is required for public authorities, large-scale systematic monitoring, and large-scale processing of special categories of data."

    print("faithfulness  (expect about 0.67):", faithfulness(answer, context))
    print("context_recall (expect about 0.67):", context_recall(expected, context))
    print("correctness   (expect 0 or 0.5)  :", correctness(answer, expected))