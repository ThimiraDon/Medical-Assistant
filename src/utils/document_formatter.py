

def format_documents(docs):

    formatted_docs = []

    for i, doc in enumerate(docs):
        formatted_docs.append(
            f"Document {i+1}:\n{doc.page_content}"
        )

    return "\n\n".join(formatted_docs)