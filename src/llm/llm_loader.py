from langchain_groq import ChatGroq
from src.config import LLM_MODEL,SMALL_MODEL
from src.config.API_config import GROQ_API_KEY


class LLMLoader:
    def __init__(self,groq_api_key):
        self.model = ChatGroq(
            model=LLM_MODEL, 
            api_key=groq_api_key,
            temperature=0
            )
        self.small_model = ChatGroq(
            model=SMALL_MODEL, 
            api_key=groq_api_key,
            temperature=0
        )

    def get_model(self):
        return self.model
    
    def get_small_model(self):
        return self.small_model