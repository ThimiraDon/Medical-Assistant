from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QueryRewriter:
    def __init__(self,llm):
        self.llm = llm
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
                        You are a medical search query optimizer.

                        Rewrite the user's question into a clear standalone medical search query.

                        Rules:
                            - Preserve the medical meaning
                            - Resolve ambiguous references using conversation context
                            - Expand abbreviations
                            - Include disease or drug names if implied
                            - Use standard medical terminology
                            - Keep the query concise
                            - Do NOT answer the question

                        Return ONLY the rewritten query.
                        """),
            ("human",                    
             """
                Conversation History: {history}

                User's Question: {query}

             """
              )   
        ])
        #StrOutputParser ensures that the output is returned as a string
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def rewrite(self, query:str, history:str="") -> str:
        rewritten_query = self.chain.invoke(
            {"query": query,
             "history": history
            }
        )
        return rewritten_query
    
