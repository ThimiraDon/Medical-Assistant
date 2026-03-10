from src.query_rewriter.Conversation_aware_rewriter import QueryRewriter
from src.query_rewriter.multi_query_gen_rewriter import MultiQueryGenerator
from src.query_rewriter.query_decomposer import QueryDecomposer


from difflib import SequenceMatcher

try:
    import spacy
    nlp = spacy.load("en_core_sci_sm")  # SciSpacy for medical entities
except:
    nlp = None 
class RewriteQueryPipeline:

    def __init__(self, llm, max_decomposed=2, max_variants=2,max_words=12, dedup_threshold=0.8):

        self.rewriter = QueryRewriter(llm)
        self.multi_query = MultiQueryGenerator(llm)
        self.decomposer = QueryDecomposer(llm)

        self.max_decomposed = max_decomposed
        self.max_variants = max_variants
        self.max_words = max_words
        self.dedup_threshold = dedup_threshold

    # Deduplicate very similar queries
    def deduplicate_queries(self, queries):
        final_queries = []
        for q in queries:
            if all(SequenceMatcher(None, q, existing).ratio() < self.dedup_threshold for existing in final_queries):
                final_queries.append(q)
        return final_queries
    
    #Truncate queries to max_words
    def smart_truncate(self, query):

        intent_words = [
            "symptoms","treatment","therapy",
            "causes","prevention","complications"
        ]

        words = query.lower().split()

        filtered = [w for w in words if w in intent_words]

        if filtered:
            return " ".join(words[:self.max_words])

        return " ".join(words[:self.max_words])

    def process(self, query: str, history: str = "") -> list:

        # Step 1: Rewrite query using conversation context
        rewritten_query = self.rewriter.rewrite(query, history)

        # Step 2: Decompose query (optional, only if complex)
        decomposed_queries = self.decomposer.decompose(rewritten_query)
        if not decomposed_queries:
            decomposed_queries = [rewritten_query]
        decomposed_queries = decomposed_queries[:self.max_decomposed]

        # Step 3: Generate multi-query variants
        final_queries = set()
        for q in decomposed_queries:
            variants = self.multi_query.generate(q)
            if not variants:
                variants = [q]
            variants = variants[:self.max_variants]

            # Step 4: Smart truncate
            variants = [self.smart_truncate(v) for v in variants]
            final_queries.update(variants)

        # Step 5: Deduplicate
        final_queries = self.deduplicate_queries(list(final_queries))

        return final_queries