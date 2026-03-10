from sentence_transformers import CrossEncoder
from src.config import RERANKER, RERANK_K

class ReRanker:
    #Cross-encoder re-ranking for retrieved documents

    def __init__(self,model_name=RERANKER,top_k=RERANK_K):
        
        self.model = CrossEncoder(model_name)
        self.top_k=top_k

    def rerank(self, query,docs):
        if not docs:
            return None
        
        pairs = [[query, doc.page_content] for doc in docs]

        scores = self.model.predict(pairs)

        ranked_docs = sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True
        )

        return [doc for score, doc in ranked_docs[:self.top_k]]