from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QueryDecomposer:

    def __init__(self, llm):

        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             """
                Break the question into smaller independent search queries
                if it contains multiple medical questions.

                If the question is simple, return the original query.

                Return each query on a new line.
            """),
            ("human", "{query}")
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def decompose(self, query: str):

        output = self.chain.invoke({"query": query})

        queries = [q.strip() for q in output.split("\n") if q.strip()]

        return queries