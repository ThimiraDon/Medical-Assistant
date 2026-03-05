#making dir
mkdir -p src
mkdir -p src/ingestion
mkdir -p research
mkdir -p src/embeddings
mkdir -p src/config
mkdir -p pipelines
mkdir -p src/prompts


#creating files
touch src/__init__.py
touch src/config/__init__.py
touch src/ingestion/__init__.py
touch src/ingestion/loader.py
touch src/ingestion/chunker.py
touch src/embeddings/__init__.py
touch src/embeddings/pinecone_loader.py
touch src/retriever.py
touch src/prompts/__init__.py
touch src/prompts/prompt_template.py
touch src/prompts/fewshot_example.py
touch src/logger.py
touch src/llm_rag.py
touch .env
touch setup.py
touch app.py
touch main.py
touch requirements.txt
touch research/test.ipynb
touch pipelines/__init__.py
touch pipelines/step_01_clean_pipeline.py
touch pipelines/step_02_chunk_pipeline.py
touch pipelines/step_03_embedding_pipeline.py



echo "All the Files Created Successfully!"