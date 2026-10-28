from src.retrieval.retrieve import retrieve

relevant = [
    "When is a DPO required?",
    "GDPR breach notification deadline",
    "What are the rights of a data subject under GDPR?",
    "What is the maximum fine for a serious GDPR violation?",
    "What are the conditions for valid consent under GDPR?",
    "What are the bookkeeping obligations under HGB?",
    "What is the retention period for accounting records under HGB?",
    "What is the definition of a merchant under HGB?",
    "What does HGB require regarding the principle of prudence?",
    "What are the requirements for an inventory under HGB?",
    "Welche Aufbewahrungsfristen gelten für Handelsbücher?",
    "Wann ist ein Datenschutzbeauftragter erforderlich?",
]
off_topic = [
    "What is the weather in Berlin?",
    "What is the capital of France?",
    "Who won the football world cup?",
    "How do I bake bread?",
    "What is the corporate tax rate in Germany?",
    "How do I reset my Wi-Fi router?",
]

for label, questions in [("RELEVANT", relevant), ("OFF-TOPIC", off_topic)]:
    print(label)
    for q in questions:
        docs, metas, scores = retrieve(q, top_k=3)
        print(f"  {scores[0]:7.2f}  [{metas[0]['source']}]  {q}")