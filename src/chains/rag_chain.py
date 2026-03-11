from langchain.schema import Document
from src.utils.document_formatter import format_documents
MAX_CHARS_PER_DOC = 1000

class MedicalRAGChain:
    def __init__(self, llm, retriever, prompt, memory, query_pipeline,reranker):
        """
        llm: LLM object
        retriever: MultiQueryRetriever instance
        prompt: prompt template
        memory: MemoryManager instance
        query_pipeline: query rewriter / multi-query generator
        """
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt
        self.memory = memory
        self.query_pipeline = query_pipeline
        self.reranker=reranker

    def run(self, user_query, debug=False):
        #Build memory context
        memory_context = self.memory.build_context(user_query)

        #Generate rewritten queries using memory context
        rewrite_queries = self.query_pipeline.process(user_query, history=memory_context)

        #Get the context from memeory
        all_docs = []
        Retrival_context=""
        for q in rewrite_queries:
            retrieve_docs = self.retriever.retrieve([q])
            docs_content = [d if hasattr(d,"page_content") else Document(page_content=str(d))
                            for d in retrieve_docs]
            all_docs.extend(docs_content)
        
        all_docs = self.retriever.deduplicate_docs(all_docs)
        #print(all_docs)
        #print("\n\n")
        if all_docs:
            # --- Re-rank documents ---
            ranked_docs = self.reranker.rerank(user_query, all_docs)
             #print(ranked_docs)
            Retrival_context="\n".join([i.page_content[:MAX_CHARS_PER_DOC] for i in ranked_docs])
            #print(Retrival_context)

        #long-term memory
        long_term_memory = ""
        if hasattr(self.memory,"vector"):
            long_term_memory=self.memory.vector.retrieve(user_query)
            
        final_context=f"""
            Relevant Medical Knowledge:
            {Retrival_context}

            Relevant Past Conversation:
            {long_term_memory}
            """
        prompt = self.prompt.get_prompt()
        formatted_prompt = prompt.format(
                context=final_context,
                input=user_query
            )
        
        #response = self.llm.invoke(formatted_prompt).content

        #Stream tokens from LLM and save
        full_response = ""
        for token in self.llm.stream(formatted_prompt):
            # Convert token to string if it's an object
            if hasattr(token, "content"):      # AIMessageChunk
                text = token.content
            else:                              # plain string fallback
                text = str(token)

            full_response += text
            yield text  # stream to console / frontend

        self.memory.store_interaction(user_query=user_query,response=full_response)

        #return response