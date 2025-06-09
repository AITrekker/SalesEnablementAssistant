"""
This module contains the core Retrieval-Augmented Generation (RAG) pipeline.

The pipeline orchestrates the following steps:
1.  **Embed**: Converts a user's query into a vector embedding.
2.  **Search**: Uses the query embedding to search the vector database for relevant document chunks.
3.  **Augment**: Combines the retrieved chunks (context) with the original query into a detailed prompt for the LLM.
4.  **Generate**: Sends the augmented prompt to an LLM to generate a final, grounded answer.
"""
import os
import ollama
from backend.vector_db import search_vector_db
from config import settings

def get_rag_response_stream(query):
    """
    Executes the RAG pipeline and returns a stream of the LLM response and the formatted source documents.

    Args:
        query (str): The user's input query.

    Returns:
        tuple: A tuple containing:
            - stream (iterator): An iterator for the LLM's response chunks.
            - sources_text (str): A formatted string of the source documents.
            Returns (None, "Error message") if no documents are found.
    """
    # === 1. EMBED: Convert the user's query into a vector embedding. ===
    query_embedding = ollama.embeddings(
        model=settings.EMBEDDING_MODEL,
        prompt=query
    )['embedding']

    # === 2. SEARCH: Retrieve relevant document chunks from the vector DB. ===
    retrieved_chunks = search_vector_db(query_embedding, top_k=5)

    if not retrieved_chunks:
        error_message = "No relevant documents were found in the database. Please try rephrasing your question or ingesting more documents."
        return None, error_message

    # === 3. AUGMENT: Format sources and construct the final prompt. ===
    
    # Format the source documents for display in the UI
    source_documents = []
    seen_paths = set()
    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {})
        source_path = metadata.get("source_path")
        if source_path and source_path not in seen_paths:
            # Use os.path.basename for a cleaner, file-only display
            source_documents.append(os.path.basename(source_path))
            seen_paths.add(source_path)
            
    sources_text = "### Retrieved Sources:\n" + "\n".join(f"- {doc}" for doc in sorted(source_documents))
    print(f"Retrieved context from: {', '.join(sorted(source_documents))}")

    # Combine the content of the retrieved chunks into a single context string
    context_str = "\n\n---\n\n".join([chunk['document'] for chunk in retrieved_chunks])
    
    # Format the final prompt using the template from settings
    final_prompt = settings.RAG_PROMPT_TEMPLATE.format(context=context_str, query=query)

    # === 4. GENERATE: Get the response stream from the LLM. ===
    llm_stream = ollama.chat(
        model=settings.LLM_MODEL,
        messages=[{'role': 'user', 'content': final_prompt}],
        stream=True
    )

    return llm_stream, sources_text