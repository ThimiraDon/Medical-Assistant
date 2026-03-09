from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class MultiQueryGenerator:

    def __init__(self, llm):

        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             """
                    Generate 4 different medical search queries
                    that could retrieve relevant documents.

                    Use synonyms and alternative phrasing.

                    Return each query on a new line.
                    """),
            ("human", "{query}")
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(self, query: str):

        output = self.chain.invoke({"query": query})

        queries = [q.strip() for q in output.split("\n") if q.strip()]

        return queries