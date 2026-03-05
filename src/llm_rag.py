from src.config import LLM_MODEL, TEMPERATURE
from src.retriever import MedicalRetriever
from src.prompts.prompt_template import MedicalPrompt

from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from dotenv import load_dotenv
import certifi

# Set environment variable so Python can find CA certificates
os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not set in .env file")

class MedicalAssistant:
    def __init__(self):
        self.retriever = MedicalRetriever().get_vectors()
        self.prompt_template = MedicalPrompt().get_prompt()
        
        try:
            self.llm = ChatGroq(
                api_key=groq_api_key,
                model=LLM_MODEL, 
                temperature=TEMPERATURE
                )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChatGroq LLM: {e}") from e

    def ask(self,query):

        # Ensure LLM is initialized before asking
        if not hasattr(self, "llm") or self.llm is None:
            raise RuntimeError("LLM is not initialized. Cannot process the query.")
        
        question_answer_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.prompt_template
            )
        
        retrieval_chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=question_answer_chain,
        )
        response = retrieval_chain.invoke({"input": query})
        return response["answer"]
    