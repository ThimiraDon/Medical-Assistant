from langchain_groq import ChatGroq
from src.config import LLM_MODEL
from src.config.API_config import GROQ_API_KEY


class LLMLoader:
    def __init__(self,groq_api_key):
        self.model = ChatGroq(
            model=LLM_MODEL, 
            api_key=groq_api_key,
            temperature=0
            )

    def get_model(self):
        return self.model