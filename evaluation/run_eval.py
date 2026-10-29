import json
import os
import sys
import time

import mlflow
import pandas as pd

# Make "src" and "evaluation" importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.judge import faithfulness, context_recall, correctness, JUDGE_MODEL
from src.analysis.rag_agent import (
    ask_with_context, llm, RELEVANCE_THRESHOLD, OUT_OF_SCOPE_MSG,
)

RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "baseline"
CSV_PATH = f"evaluation/eval_per_question_{RUN_NAME}.csv"
MD_PATH = f"evaluation/eval_results_{RUN_NAME}.md"
METRICS = ["faithfulness", "context_recall", "correctness"]

with open("evaluation/test_queries.json", encoding="utf-8") as f:
    items = json.load(f)

rows = []
for i, item in enumerate(items, start=1):
    print(f"[{i}/{len(items)}] {item['question'][:65]}")
    try:
        r = ask_with_context(item["question"])
    except Exception as e:
        print("   ERROR while answering:", str(e)[:100])
        continue

    answer = r["answer"]
    context = "\n\n".join(r["contexts"])
    refused = OUT_OF_SCOPE_MSG.lower().rstrip(".") in answer.lower()

    # A refusal on an in-scope question is a failure; skip the judge for it
    faith = None if refused else faithfulness(answer, context)
    recall = 0.0 if not r["contexts"] else context_recall(item["expected_answer"], context)
    correct = 0.0 if refused else correctness(answer, item["expected_answer"])

    rows.append({
        "question": item["question"],
        "regulation": item["regulation"],
        "refused": refused,
        "faithfulness": faith,
        "context_recall": recall,
        "correctness": correct,
        "top_score": r["top_score"],
        "latency_s": round(r["latency"], 1),
        "answer": answer,
    })
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False, encoding="utf-8")  # save as we go
    print(f"   faith={faith}  recall={recall}  correct={correct}  refused={refused}")
    time.sleep(1)

df = pd.DataFrame(rows)
means = df[METRICS].mean()
by_reg = df.groupby("regulation")[METRICS].mean()
median_latency = df["latency_s"].median()
refusals = int(df["refused"].sum())

print("\n=== BASELINE RESULTS ===")
print(means.round(3).to_string())
print("\nBy regulation:")
print(by_reg.round(3).to_string())
print(f"\nMedian latency: {median_latency:.1f}s   Refusals of real questions: {refusals}")
print("\nWeakest questions (lowest correctness):")
print(df.sort_values("correctness")[["question", "correctness", "context_recall"]].head(5).to_string(index=False))

# ---- MLflow ----
mlflow.set_experiment("compliance-rag-p1")
with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_params({
        "chunk_size": 512,
        "chunk_overlap": 50,
        "chunking": "article/section-aware with heading prefix",
        "embedding_model": "intfloat/multilingual-e5-large",
        "reranker_model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "llm_model": getattr(llm, "model_name", "unknown"),
        "judge_model": JUDGE_MODEL,
        "chunks_to_llm": 5,
        "same_article_neighbours": True,
        "glossary_terms": "DPO, DPIA, DSAR (HGB entry removed)",
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "num_questions": len(df),
    })
    metrics = {k: float(v) for k, v in means.items() if pd.notna(v)}
    metrics["median_latency_s"] = float(median_latency)
    metrics["refusals"] = refusals
    for reg, row in by_reg.iterrows():
        for m in METRICS:
            if pd.notna(row[m]):
                metrics[f"{m}_{reg}"] = float(row[m])
    mlflow.log_metrics(metrics)

    # ---- Markdown report ----
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(f"# RAG Evaluation Results: {RUN_NAME}\n\n")
        f.write("Metrics are RAGAs-style LLM-as-judge scores (0 to 1), implemented with Groq.\n\n")
        f.write("## Settings\n")
        f.write(f"- Answer model: {getattr(llm, 'model_name', 'unknown')}\n")
        f.write(f"- Judge model: {JUDGE_MODEL}\n")
        f.write("- Embeddings: intfloat/multilingual-e5-large\n")
        f.write("- Reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1\n")
        f.write("- Chunking: article/section-aware, 512 chars, 50 overlap, heading prefix\n")
        f.write(f"- Relevance threshold: {RELEVANCE_THRESHOLD}\n\n")
        f.write("## Scores (average over all questions)\n")
        for k, v in means.items():
            f.write(f"- {k}: {v:.3f}\n")
        f.write(f"- median latency: {median_latency:.1f}s\n")
        f.write(f"- real questions refused: {refusals}\n\n")
        f.write("## By regulation\n\n")
        f.write(by_reg.round(3).to_markdown() + "\n\n")
        f.write("## Per question\n\n")
        short = df[["question", "regulation", "faithfulness", "context_recall", "correctness"]].copy()
        short["question"] = short["question"].str.slice(0, 60)
        f.write(short.round(2).to_markdown(index=False) + "\n")

    mlflow.log_artifact(CSV_PATH)
    mlflow.log_artifact(MD_PATH)

print(f"\nSaved {CSV_PATH} and {MD_PATH}. MLflow run '{RUN_NAME}' logged.")