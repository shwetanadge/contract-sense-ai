from src.retrieval.retrieve import retrieve

relevant = [
    "When is a DPO required?",
    "GDPR breach notification deadline",
    "HGB bookkeeping obligations",
    "What are HGB payment terms?",
    "What are the rights of a data subject under GDPR?",
]
off_topic = [
    "What is the weather in Berlin?",
    "What is the capital of France?",
    "Who won the football world cup?",
    "How do I bake bread?",
]

for label, questions in [("RELEVANT", relevant), ("OFF-TOPIC", off_topic)]:
    print(label)
    for q in questions:
        _, _, scores = retrieve(q, top_k=3)
        print(f"  {scores[0]:7.2f}  {q}")