# RAG Evaluation Results: improvement_1

Metrics are RAGAs-style LLM-as-judge scores (0 to 1), implemented with Groq.

## Settings
- Answer model: openai/gpt-oss-20b
- Judge model: openai/gpt-oss-120b
- Embeddings: intfloat/multilingual-e5-large
- Reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
- Chunking: article/section-aware, 512 chars, 50 overlap, heading prefix
- Relevance threshold: -1.0

## Scores (average over all questions)
- faithfulness: 0.947
- context_recall: 0.723
- correctness: 0.675
- median latency: 4.8s
- real questions refused: 1

## By regulation

| regulation   |   faithfulness |   context_recall |   correctness |
|:-------------|---------------:|-----------------:|--------------:|
| GDPR         |          1     |            0.933 |          0.95 |
| HGB          |          0.889 |            0.513 |          0.4  |

## Per question

| question                                                     | regulation   |   faithfulness |   context_recall |   correctness |
|:-------------------------------------------------------------|:-------------|---------------:|-----------------:|--------------:|
| When must a Data Protection Officer be appointed?            | GDPR         |              1 |             1    |           1   |
| What is the deadline for reporting a personal data breach to | GDPR         |              1 |             1    |           0.5 |
| What are the lawful bases for processing personal data under | GDPR         |              1 |             1    |           1   |
| What rights does a data subject have regarding their persona | GDPR         |              1 |             0.67 |           1   |
| What is the maximum fine for a serious GDPR violation?       | GDPR         |              1 |             1    |           1   |
| What is the right to erasure under GDPR?                     | GDPR         |              1 |             1    |           1   |
| What is a Data Protection Impact Assessment and when is it r | GDPR         |              1 |             1    |           1   |
| What information must be provided to data subjects when coll | GDPR         |              1 |             1    |           1   |
| What is the definition of personal data under GDPR?          | GDPR         |              1 |             1    |           1   |
| What are the conditions for valid consent under GDPR?        | GDPR         |              1 |             0.67 |           1   |
| What is the retention period for accounting records under HG | HGB          |              1 |             0.33 |           0.5 |
| What is the obligation of a merchant regarding bookkeeping u | HGB          |              1 |             1    |           1   |
| What is the deadline for preparing annual financial statemen | HGB          |              0 |             0    |           0   |
| What does HGB require regarding the principle of prudence?   | HGB          |              1 |             0    |           0   |
| What is the going concern principle under HGB?               | HGB          |            nan |             1    |           0   |
| What are the requirements for an inventory under HGB?        | HGB          |              1 |             1    |           0.5 |
| What is the definition of a merchant under HGB?              | HGB          |              1 |             1    |           1   |
| What does HGB require regarding the consistency principle?   | HGB          |              1 |             0    |           0   |
| What is the obligation regarding commercial letters under HG | HGB          |              1 |             0    |           0.5 |
| What information must be included in commercial letters unde | HGB          |              1 |             0.8  |           0.5 |
