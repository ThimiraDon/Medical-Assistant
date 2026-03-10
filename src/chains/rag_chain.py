from langchain.schema import Document
from src.utils.document_formatter import format_documents

class MedicalRAGChain:
    def __init__(self, llm, retriever, prompt, memory, query_pipeline):
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
        Retrival_context="\n".join([i.page_content for i in all_docs])
        #print(Retrival_context)

        #long-term memory
        long_term_memory = ""
        if hasattr(self.memory,"vector"):
            long_term_memory=self.memory.vector.retrieve(user_query)
            
        # --- Fallback for empty retrieval ---
        if not all_docs:
            # Get user's name from EntityMemory
            user_name = self.memory.entities.get_entities(category="NAME")
            if user_name:
                # Take first name if multiple
                user_name = list(user_name.split(":")[1].strip().split(","))[0]
                response = f"I’m not familiar with that topic yet, {user_name}. Could you clarify or ask a different question?"
            else:
                response = "I'm not sure about that yet, but I can help you find information. Can you tell me more?"

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
        
        response = self.llm.invoke(formatted_prompt).content

        self.memory.store_interaction(user_query=user_query,response=response)

        return response