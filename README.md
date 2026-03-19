# Website RAG Assistant
I setup ingestion and query pipeline RAG chatbbot, so users to get information on my website I setup a RAG assistent to answer queries from select sources (uploaded resume, portfolio website, blogs, linkedin).

## Ingestion-Query Pipeline Flow
![alt text](assets/images/website-rag-ingest-query-pp.png)
### Ingestion Pipeline
### Manual Upload (single files)
- Chainlit/Interface
  - User uploads source docs using chainlit interface
  - Intent router decides if upload or query
  - upload docs
  - embeddings, chunking - using openai-embedding-3-large model
  - save to vector DB - qdrant cloud
### Bulk Upload (multiple sources)
- Setup via cron jobs
- Ingests external sources
  - substack blogs
  - portfolio website
  - linkedin profile
### Query Pipeline
Ask questions via interface
- Query
- Intent classifier (upload/ query)
- Embdding
- Cosine search
- 

## Demo
Query, Citations
![alt text](assets/images/query-citations-1.png)


### Production Ready Features
- Obervability
  - Traces, Latency p50, P90
  - Inbuilt heuristic metrics
  - LLM as a judge metrics
- Evaluation, with golden datasets
- Central prompt management, versioning
- Evals on every commit w pre commit hooks, and skippable flags

## Evals on every commit w pre commit hooks, and skippable flags
![alt text](assets/images/evals%20on%20commit%20w%20pre%20commit%20hooks.png)

## Traces Latency Overall
![alt text](assets/images/tracing%20latency%20overall%20dashboard.png)
## Golden Dataset
![alt text](assets/images/golden-dataset.png)

## Central Prompt Management, Versioning
![alt text](assets/images/prompt-mgmt-query.png)

## Prompt Versioning
![alt text](assets/images/prompt-versioning-intent.png)

## Evaluations against golden dataset
![alt text](assets/images/evals-exps-vs-golden-dataset.png)


## Feedbacks using rule-based, LLM as a judge metrics
![alt text](assets/images/basic-rag-metrics-w-llm-as-judge-feedback.png)


## Optimisations
- HNSW configurations for faster embedding
- Faster inferences using Groq
- Open source models to minimise costs

<!-- Prompt Versioning
![alt text](assets/images/.png) -->

