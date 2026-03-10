import os
os.environ["SSL_CERT_FILE"] = r"C:\certs\cacert.pem"

from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

from src.config.API_config import GROQ_API_KEY
from src.llm.llm_loader import LLMLoader
from src.prompts.prompt_template import MedicalPrompt
from src.memory.memory_manager import MemoryManager
from src.retriever.retriever import MultiQueryRetriever
from src.query_rewriter.rewrite_query_pipeline import RewriteQueryPipeline
from src.reranker.reranking import ReRanker

from src.chains.rag_chain import MedicalRAGChain

from src.utils.document_formatter import format_response_html


app=Flask(__name__)

load_dotenv()
llm =LLMLoader(groq_api_key=GROQ_API_KEY).get_model()
small_llm=LLMLoader(groq_api_key=GROQ_API_KEY).get_small_model()

#prompt_template
prompt = MedicalPrompt()

#initialize memory manager
memeory_manager = MemoryManager(llm=small_llm)
memeory_manager.reset_memory()

#Initialize query pipeline (rewriter + multi-query + smart truncate)
query_pipeline=RewriteQueryPipeline(llm=small_llm)

#Initialize retrieve
retriever=MultiQueryRetriever(llm=small_llm,memory_manager=memeory_manager)

#Initialize ReRanker
reranker = ReRanker(top_k=3)

rag_chain = MedicalRAGChain(
                llm=llm,
                retriever=retriever,
                prompt=prompt,
                memory=memeory_manager,
                query_pipeline=query_pipeline,
                reranker=reranker
            )


@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]

    print("User:", msg)

    response = rag_chain.run(msg)
    formatteed_response=format_response_html(response)
    print("Bot:", formatteed_response)
    return str(formatteed_response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)