import os
import time
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from src.retrieval.retrieve import retrieve
from src.storage.db import log_query

load_dotenv()

# Define agent state
class AgentState(TypedDict):
    question: str
    retrieved_chunks: list
    answer: str
    citations: list

# Initialize LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

# Node 1 - Retrieve
def retrieve_node(state: AgentState):
    question = state["question"]
    docs, metas, scores = retrieve(question, top_k=3)
    chunks = [{"text": doc, "metadata": meta} 
              for doc, meta in zip(docs, metas)]
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
Cite sources as [1], [2], [3].
If the answer is not in the excerpts, say exactly: "This question is outside the scope of the loaded regulations."

Regulation excerpts:
{context}

Question: {question}

Answer:"""
    
    response = llm.invoke(prompt)
    answer = response.content
    
    return {"answer": answer, "citations": citations}

# Build the graph
def build_graph():
    graph = StateGraph(AgentState)
    
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    
    graph.set_entry_point("retrieve")
    
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