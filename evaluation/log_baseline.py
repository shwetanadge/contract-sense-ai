import os
import sys

import mlflow
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.judge import JUDGE_MODEL
from src.analysis.rag_agent import llm, RELEVANCE_THRESHOLD

RUN_NAME = "baseline"
CSV_PATH = "evaluation/eval_per_question.csv"
MD_PATH = "evaluation/eval_results.md"
METRICS = ["faithfulness", "context_recall", "correctness"]

df = pd.read_csv(CSV_PATH)
means = df[METRICS].mean()
by_reg = df.groupby("regulation")[METRICS].mean()
median_latency = float(df["latency_s"].median())
refusals = int(df["refused"].sum())
model_name = getattr(llm, "model_name", "unknown")

with open(MD_PATH, "w", encoding="utf-8") as f:
    f.write(f"# RAG Evaluation Results: {RUN_NAME}\n\n")
    f.write("Metrics are RAGAs-style LLM-as-judge scores (0 to 1), implemented with Groq.\n\n")
    f.write("## Settings\n")
    f.write(f"- Answer model: {model_name}\n- Judge model: {JUDGE_MODEL}\n")
    f.write("- Embeddings: intfloat/multilingual-e5-large\n")
    f.write("- Reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1\n")
    f.write("- Chunking: article/section-aware, 512 chars, 50 overlap, heading prefix\n")
    f.write(f"- Relevance threshold: {RELEVANCE_THRESHOLD}\n\n")
    f.write("## Scores (average over all questions)\n")
    for k, v in means.items():
        f.write(f"- {k}: {v:.3f}\n")
    f.write(f"- median latency: {median_latency:.1f}s\n")
    f.write(f"- real questions refused: {refusals}\n\n")
    f.write("## By regulation\n\n" + by_reg.round(3).to_markdown() + "\n\n")
    f.write("## Per question\n\n")
    short = df[["question", "regulation", "faithfulness", "context_recall", "correctness"]].copy()
    short["question"] = short["question"].str.slice(0, 60)
    f.write(short.round(2).to_markdown(index=False) + "\n")

mlflow.set_experiment("compliance-rag-p1")
with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_params({
        "chunk_size": 512, "chunk_overlap": 50,
        "chunking": "article/section-aware with heading prefix",
        "embedding_model": "intfloat/multilingual-e5-large",
        "reranker_model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "llm_model": model_name, "judge_model": JUDGE_MODEL,
        "chunks_to_llm": 5, "same_article_neighbours": True,
        "query_expansion": True, "relevance_threshold": RELEVANCE_THRESHOLD,
        "num_questions": len(df),
    })
    metrics = {k: float(v) for k, v in means.items()}
    metrics["median_latency_s"] = median_latency
    metrics["refusals"] = refusals
    for reg, row in by_reg.iterrows():
        for m in METRICS:
            metrics[f"{m}_{reg}"] = float(row[m])
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(CSV_PATH)
    mlflow.log_artifact(MD_PATH)

print("Baseline logged to MLflow and report written to", MD_PATH)