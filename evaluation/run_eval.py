import json
import time
import mlflow
import os
from dotenv import load_dotenv
from src.ingestion.pdf_parser import extract_text
from src.ingestion.chunker import chunk_text
from src.retrieval.vector_store import load_bm25
from src.analysis.rag_agent import ask

load_dotenv()

# Load BM25
print("Loading BM25 index...")
text = extract_text("data/sample_contracts/gdpr.pdf")
chunks = chunk_text(text, "GDPR")
load_bm25(chunks)

# Load test questions
with open("evaluation/test_queries.json", "r") as f:
    test_queries = json.load(f)

# Run evaluation
print(f"Running evaluation on {len(test_queries)} questions...")

questions = []
answers = []
expected_answers = []
contexts = []

for i, item in enumerate(test_queries):
    print(f"Question {i+1}/{len(test_queries)}: {item['question'][:50]}...")
    
    try:
        answer, citations = ask(item['question'])
        questions.append(item['question'])
        answers.append(answer)
        expected_answers.append(item['expected_answer'])
        contexts.append(citations)
        time.sleep(1)  # avoid rate limiting
    except Exception as e:
        print(f"Error on question {i+1}: {e}")
        continue

print(f"\nSuccessfully evaluated {len(questions)} questions")

# MLflow tracking
mlflow.set_experiment("compliance-rag-p1")

with mlflow.start_run(run_name="baseline"):
    # Log parameters
    mlflow.log_param("chunk_size", 512)
    mlflow.log_param("chunk_overlap", 50)
    mlflow.log_param("embedding_model", "multilingual-e5-large")
    mlflow.log_param("llm_model", "llama-3.3-70b-versatile")
    mlflow.log_param("top_k", 3)
    mlflow.log_param("num_questions", len(questions))

    # Calculate simple scores manually
    # (avoiding RAGAs OpenAI dependency)
    
    # Score 1: Answer length score (proxy for completeness)
    avg_answer_length = sum(len(a) for a in answers) / len(answers)
    completeness_score = min(avg_answer_length / 500, 1.0)
    
    # Score 2: Citation rate (how often citations appear)
    cited_answers = sum(1 for a in answers if '[' in a)
    citation_rate = cited_answers / len(answers)
    
    # Score 3: Guardrail accuracy
    # Question 11 onwards are HGB - check if answered
    answered_count = sum(1 for a in answers 
                        if 'outside the scope' not in a)
    answer_rate = answered_count / len(answers)
    
    # Log metrics
    mlflow.log_metric("citation_rate", round(citation_rate, 3))
    mlflow.log_metric("answer_rate", round(answer_rate, 3))
    mlflow.log_metric("completeness_score", round(completeness_score, 3))
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Citation Rate: {citation_rate:.3f}")
    print(f"Answer Rate: {answer_rate:.3f}")
    print(f"Completeness Score: {completeness_score:.3f}")
    
    # Save results to markdown
    with open("evaluation/eval_results.md", "w", encoding="utf-8") as f:
        f.write("# RAG Evaluation Results — Baseline\n\n")
        f.write("## Parameters\n")
        f.write("- Chunk size: 512\n")
        f.write("- Chunk overlap: 50\n")
        f.write("- Embedding model: multilingual-e5-large\n")
        f.write("- LLM: llama-3.1-8b-instant\n")
        f.write("- Top K: 3\n\n")
        f.write("## Scores\n")
        f.write(f"- Citation Rate: {citation_rate:.3f}\n")
        f.write(f"- Answer Rate: {answer_rate:.3f}\n")
        f.write(f"- Completeness Score: {completeness_score:.3f}\n\n")
        f.write("## Sample Results\n\n")
        for i, (q, a) in enumerate(zip(questions[:5], answers[:5])):
            f.write(f"### Q{i+1}: {q}\n")
            f.write(f"**Answer:** {a[:300]}...\n\n")
    
    print("\nResults saved to evaluation/eval_results.md")
    print("MLflow run complete!")