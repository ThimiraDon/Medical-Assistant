#making dir
mkdir -p src
mkdir -p src/ingestion
mkdir -p research
mkdir -p src/embeddings
mkdir -p src/config

#creating files
touch src/__init__.py
touch src/config/__init__.py
touch src/ingestion/__init__.py
touch src/ingestion/loader.py
touch src/ingestion/chunker.py
touch src/embeddings/__init__.py
touch src/embeddings/pinecone_loader.py
touch src/helper.py
touch src/prompt.py
touch src/utils.py
touch src/logger.py
touch .env
touch setup.py
touch app.py
touch main.py
touch requirements.txt
touch research/test.ipynb


echo "All the Files Created Successfully!"