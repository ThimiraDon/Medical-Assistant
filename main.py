import os
os.environ["SSL_CERT_FILE"] = r"C:\certs\cacert.pem"

from dotenv import load_dotenv
load_dotenv()

from src.config.API_config import GROQ_API_KEY
from src.llm.llm_loader import LLMLoader
from src.memory.memory_manager import MemoryManager
from src.retriever.retriever import MultiQueryRetriever
from src.query_rewriter.rewrite_query_pipeline import RewriteQueryPipeline
from src.prompts.prompt_template import MedicalPrompt
from src.reranker.reranking import ReRanker
# Optional: your RAG chain if you still want to use it
from src.chains.rag_chain import MedicalRAGChain

def main():
    print("Initializing Medical Assistant...")

    #Load the LLM
    llm = LLMLoader(groq_api_key=GROQ_API_KEY).get_model()
    #medi_chain = MedicalRAGChain()
    prompt_template = MedicalPrompt()

    #Initialize memory manager
    memory_manager = MemoryManager(llm=llm)
    memory_manager.reset_memory()

    #Initialize query pipeline (rewriter + multi-query + smart truncate)
    query_pipeline = RewriteQueryPipeline(llm=llm)

    #Initialize retriever
    retriever = MultiQueryRetriever(llm=llm, memory_manager=memory_manager)

    #Initialize ReRanker
    reranker = ReRanker(top_k=3)

    print("Medical Assistant Ready! Type 'exit' to quit.\n")

    while True:
        # Get user input
        user_query = input("User: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break

        rag_chain = MedicalRAGChain(
                llm=llm,
                retriever=retriever,
                prompt=prompt_template,
                memory=memory_manager,
                query_pipeline=query_pipeline,
                reranker=reranker
            )

        response = rag_chain.run(user_query)
        print(response)
        

if __name__ == "__main__":
    main()