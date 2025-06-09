# ingest_docs.py

"""
This module handles the ingestion of HTML documentation into the vector database.

It provides functionalities to:
- Recursively find HTML files in a given directory.
- Parse and clean the HTML content to extract the main text and title.
- Split the extracted text into manageable chunks.
- Generate vector embeddings for each chunk using an Ollama model.
- Store the chunks, their embeddings, and relevant metadata in a ChromaDB collection.

The primary function for this process is `ingest_directory()`.
The module also provides a Gradio UI component factory for the main app.
"""

import os
from bs4 import BeautifulSoup
from uuid import uuid4
import chromadb
from ollama import embeddings
import gradio as gr

from config.settings import VECTOR_DB_PATH, COLLECTION_NAME, CHUNK_SIZE_TOKENS, EMBEDDING_MODEL, DOCS_DIRECTORY

def clean_html(html_content):
    """
    Parses HTML content, removes script/style tags, and extracts the title and clean text.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    title = soup.title.string.strip() if soup.title else "Untitled Document"
    text = soup.get_text(separator="\n", strip=True)
    return title, text

def chunk_text(text, max_tokens=CHUNK_SIZE_TOKENS):
    """
    Splits a long text into smaller chunks based on a token limit.
    This implementation is a simple paragraph-based splitter.
    """
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk.split()) + len(para.split()) < max_tokens:
            current_chunk += para + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def ingest_directory(folder_path):
    """
    Processes all HTML files in a directory, chunks their content, and stores
    them as embeddings in the ChromaDB vector database.
    """
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    summary = []
    print(f"--- Starting ingestion from: {folder_path} ---")
    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.lower().endswith((".html", ".htm")):
                continue

            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()

            try:
                title, text = clean_html(html)
                chunks = chunk_text(text)
                print(f"  → Processing {file}: Found {len(chunks)} chunks.")

                if not chunks:
                    continue

                # Generate embeddings for each chunk individually
                doc_embeddings = []
                for chunk in chunks:
                    response = embeddings(model=EMBEDDING_MODEL, prompt=chunk)
                    doc_embeddings.append(response['embedding'])

                # Prepare data for ChromaDB
                collection.add(
                    ids=[str(uuid4()) for _ in chunks],
                    embeddings=doc_embeddings,
                    documents=chunks,
                    metadatas=[{
                        "title": title,
                        "source_path": file_path
                    } for _ in chunks]
                )
                summary.append(f"✓ {file} — {len(chunks)} chunks indexed")
            except Exception as e:
                msg = f"❌ Failed to process {file_path}: {e}"
                print(msg)
                summary.append(msg)

    print("--- ✅ Ingestion complete. Vector DB has been updated. ---")
    return "\n".join(summary)

def create_ingest_ui_components():
    """
    Creates and returns the set of Gradio components for the ingestion UI tab.
    This allows the main app to flexibly manage the layout.
    """
    folder_path_input = gr.Textbox(
        label="Folder Path to HTML Docs",
        value=DOCS_DIRECTORY,
        info="Enter the local folder path containing your HTML documentation."
    )
    ingestion_log_output = gr.Textbox(
        label="Ingestion Log",
        lines=20,
        interactive=False,
        placeholder="Ingestion progress will be shown here..."
    )
    return folder_path_input, ingestion_log_output

def handle_folder(folder_path):
    """Simple wrapper to trigger the ingestion process from the Gradio UI."""
    if not os.path.isdir(folder_path):
        return "Error: Invalid folder path. Please check the path and try again."
    return ingest_directory(folder_path)
