"""
This module provides an abstraction layer for interacting with the vector database.

Its primary responsibility is to handle similarity searches against the ChromaDB
collection. It takes a query embedding and returns the most relevant document
chunks based on vector similarity.
"""
import chromadb
from config.settings import VECTOR_DB_PATH, COLLECTION_NAME
import os

def search_vector_db(query_embedding, top_k=5):
    """
    Performs a similarity search in the ChromaDB collection.

    Args:
        query_embedding (list): The embedding vector of the user's query.
        top_k (int): The number of top results to retrieve.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains
                    a retrieved 'document' chunk and its 'metadata'.
                    Returns an empty list if the collection is not found,
                    is empty, or an error occurs.
    """
    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"]
        )

        # The `query` method returns results for a batch of queries. Since we
        # only send one query, we work with the first element of each list.
        # We also check if any IDs were returned to ensure the result is not empty.
        if results and results.get('ids') and results['ids'][0]:
            retrieved_docs = []
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                retrieved_docs.append({"document": doc, "metadata": meta})
            return retrieved_docs

        return []

    except ValueError:
        # This error is often raised by ChromaDB if the collection does not exist.
        print(f"Warning: Collection '{COLLECTION_NAME}' not found. Please ingest documents first.")
        return []
    except Exception as e:
        print(f"An error occurred during vector search: {e}")
        return []

def get_db_inspection_report():
    """
    Connects to the ChromaDB and returns a summary of its contents for display.
    """
    report = []
    if not os.path.exists(VECTOR_DB_PATH):
        return f"❌ Database directory not found at: '{VECTOR_DB_PATH}'.\nHave you ingested any documents yet?"

    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
        except ValueError:
            return f"❌ Collection '{COLLECTION_NAME}' not found in the database. Please ingest documents first."

        count = collection.count()
        report.append(f"✅ Found collection '{COLLECTION_NAME}' with {count} embedded chunks.")

        if count > 0:
            report.append("\n--- Sample of Stored Chunks (up to 5) ---")
            items = collection.get(limit=5, include=["metadatas", "documents"])
            for i, (metadata, document) in enumerate(zip(items['metadatas'], items['documents'])):
                report.append(f"\nItem #{i+1}")
                report.append(f"  Source: {os.path.basename(metadata.get('source_path', 'N/A'))}")
                report.append(f"  Title: {metadata.get('title', 'N/A')}")
                snippet = document[:150].replace('\\n', ' ').strip()
                report.append(f"  Snippet: '{snippet}...'")

        return "\\n".join(report)
    except Exception as e:
        return f"An error occurred while inspecting the database: {e}"


def clear_database_collection():
    """
    Deletes all items from the collection in ChromaDB and returns a status message.
    """
    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        # Deleting a collection is destructive. Let's ensure it exists first.
        try:
            client.get_collection(name=COLLECTION_NAME) # This will raise ValueError if it doesn't exist
            client.delete_collection(name=COLLECTION_NAME)
            # Re-create the empty collection so the app can continue to use it
            client.create_collection(name=COLLECTION_NAME)
            return f"✅ Collection '{COLLECTION_NAME}' has been cleared successfully. It now contains 0 items."
        except ValueError:
            return f"ℹ️ Collection '{COLLECTION_NAME}' did not exist, so there was nothing to clear."
    except Exception as e:
        return f"❌ An error occurred while clearing the database: {e}"