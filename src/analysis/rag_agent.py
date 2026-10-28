import os
import time
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from src.retrieval.retrieve import retrieve
from src.storage.db import log_query

load_dotenv()

RELEVANCE_THRESHOLD = -1.0   # measured on the multilingual reranker
OUT_OF_SCOPE_MSG = "This question is outside the scope of the loaded regulations."

# Define agent state
class AgentState(TypedDict):
    question: str
    retrieved_chunks: list
    answer: str
    citations: list

# Initialize LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b"
)

# Node 1 - Retrieve
def retrieve_node(state: AgentState):
    question = state["question"]
    docs, metas, scores = retrieve(question, top_k=5, add_siblings=True)
    chunks = [
        {"text": doc, "metadata": meta, "score": score}
        for doc, meta, score in zip(docs, metas, scores)
    ]
    return {"retrieved_chunks": chunks}

# Node 2 - Generate
def generate_node(state: AgentState):
    question = state["question"]
    chunks = state["retrieved_chunks"]
    
    # Build context with citations
    context = ""
    citations = []
    for i, chunk in enumerate(chunks):
        context += f"[{i+1}] {chunk['text']}\n\n"
        citations.append(f"[{i+1}] {chunk['metadata']['source']}, chunk {chunk['metadata']['chunk_id']}")
    
    # Build prompt
    prompt = f"""You are a compliance assistant. Answer using ONLY the provided regulation excerpts below.
Cite sources as [1], [2], etc., using only the numbers of excerpts you actually used.

If the answer is not in the excerpts, say exactly: "This question is outside the scope of the loaded regulations."

Regulation excerpts:
{context}

Question: {question}

Answer:"""
    
    response = llm.invoke(prompt)
    answer = response.content

    # A refusal should not show sources
    if OUT_OF_SCOPE_MSG.lower().rstrip(".") in answer.lower():
        citations = []

    return {"answer": answer, "citations": citations}

def route_after_retrieval(state: AgentState):
    chunks = state["retrieved_chunks"]
    if not chunks or chunks[0]["score"] < RELEVANCE_THRESHOLD:
        return "out_of_scope"
    return "generate"

def out_of_scope_node(state: AgentState):
    return {"answer": OUT_OF_SCOPE_MSG, "citations": []}

# Build the graph
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("out_of_scope", out_of_scope_node)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {"generate": "generate", "out_of_scope": "out_of_scope"},
    )
    graph.add_edge("generate", END)
    graph.add_edge("out_of_scope", END)

    return graph.compile()

# Compile app
app = build_graph()

# Main ask function
def ask(question):
    start = time.time()
    
    result = app.invoke({"question": question})
    
    latency = time.time() - start
    answer = result["answer"]
    citations = result["citations"]
    
    # Log to SQLite
    log_query(question, answer, latency)
    
    return answer, citations