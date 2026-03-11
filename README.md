# Memory-Aware Medical RAG Assistant

A context-aware medical AI assistant leveraging RAG, hybrid memory, and LLMs.

A **Retrieval-Augmented Generation (RAG) medical assistant** that combines advanced query rewriting, hybrid memory architecture, document re-ranking, and long-term semantic memory to deliver accurate and context-aware responses to medical questions.

The system retrieves medical knowledge from a vector database, improves retrieval through multi-query optimization, re-ranks results using a cross-encoder model, and generates responses using a large language model.

---

# 🚀 Features

### 1. Advanced Query Optimization

The assistant improves retrieval accuracy using a **multi-stage query rewriting pipeline**:

* Conversation-aware query rewriting
* Query decomposition for complex questions
* Multi-query generation using alternative phrasing
* Query deduplication and truncation

This significantly improves retrieval recall compared to standard RAG systems.

---

### 2. Hybrid Memory Architecture

The assistant maintains contextual awareness through **four complementary memory types**:

**Conversation Memory**

* Short-term conversation history
* Maintains the last few dialogue turns

**Entity Memory**

* Extracts and stores important entities such as:

  * diseases
  * drugs
  * symptoms
  * procedures
  * user identity

**Episodic Memory**

* Stores key interaction events during the conversation

**Vector Memory**

* Long-term semantic memory stored in a vector database
* Allows retrieval of relevant past interactions

A **Memory Gate** powered by an LLM decides whether interactions should be stored as long-term memory.

---

### 3. Retrieval-Augmented Generation (RAG)

The assistant retrieves medical knowledge from a vector database and combines it with conversation context before generating responses.

Key steps:

1. Query rewriting
2. Multi-query document retrieval
3. Similarity filtering
4. Cross-encoder re-ranking
5. Context assembly
6. Response generation

---

### 4. Document Re-Ranking

Retrieved documents are re-ranked using a **cross-encoder model** to ensure the most relevant medical information is passed to the LLM.

This significantly improves answer quality compared to raw vector similarity search.

---

### 5. Streaming Responses

The system streams tokens in real time, enabling responsive interaction in web interfaces or chat applications.

---

# 🏗 System Architecture

```
User Question
      │
      ▼
Memory Context Builder
      │
      ▼
Query Rewriting Pipeline
      │
      ▼
Multi-Query Retrieval (Vector DB)
      │
      ▼
Similarity Filtering
      │
      ▼
Cross-Encoder Re-Ranking
      │
      ▼
Context Construction
      │
      ▼
LLM Response Generation
      │
      ▼
Streaming Answer
      │
      ▼
Memory Update
```

---

# 🧩 Project Structure

```
src
│
├── config
│   ├── config.py
│   └── API_config.py
│
├── llm
│   └── llm_loader.py
│
├── prompts
│   ├── prompt.py
│   └── fewshot_example.py
│
├── query_rewriter
│   ├── conversation_aware_rewriter.py
│   ├── multi_query_gen_rewriter.py
│   ├── query_decomposer.py
│   └── rewrite_query_pipeline.py
│
├── memory
│   ├── conversation_memory.py
│   ├── entity_memory.py
│   ├── episodic_memory.py
│   ├── vector_memory.py
│   ├── memory_gate.py
│   └── memory_manager.py
│
├── retriever
│   └── retriever.py
│
├── reranker
│   └── reranker.py
│
└── rag
    └── rag_chain.py
```

---

# ⚙️ Technologies Used

* Python
* LangChain
* Pinecone (Vector Database)
* Sentence Transformers
* Groq LLM API
* Cross-Encoder Re-Ranking
* SciSpaCy for medical entity recognition

---

# 📊 Key Techniques Implemented

* Retrieval-Augmented Generation (RAG)
* Multi-Query Retrieval
* Query Rewriting
* Cross-Encoder Re-Ranking
* Hybrid Memory Systems
* Long-Term Vector Memory
* Context-Aware Prompt Engineering
* Streaming LLM responses

---

# 💡 Example Workflow

User Question:

```
What are the symptoms of kidney stones?
```

System Process:

1. Query rewriting generates optimized search queries
2. Vector database retrieves relevant document chunks
3. Documents are re-ranked using a cross-encoder model
4. Context is combined with conversation memory
5. The LLM generates a medically structured response

---

# 📌 Future Improvements

Possible improvements include:

* Knowledge graph integration
* Medical citation verification
* Retrieval evaluation metrics
* Multi-modal medical data support
* Agent-based reasoning workflows

---

# 📜 Disclaimer

This assistant provides **informational medical responses** and should not be considered a substitute for professional medical advice.

Always consult a qualified healthcare professional for medical concerns.

---

# 👨‍💻 Author

Developed as part of a project exploring **advanced RAG systems, memory architectures, and medical knowledge retrieval using large language models**.



