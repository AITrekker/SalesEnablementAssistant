"""
Centralized configuration settings for the RAG-based Sales Enablement Assistant.

This file defines all the critical parameters for the application, including:
- Filesystem paths for documentation and the vector database.
- Names of the Ollama models for embeddings and chat generation.
- The prompt template used to instruct the LLM in the RAG pipeline.
"""
import os

# --- PATHS ---
# Base directory of the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the folder containing the source HTML documentation.
# The ingestion script will recursively search this directory for .html and .htm files.
DOCS_DIRECTORY = os.path.join(PROJECT_ROOT, 'data', 'local_docs')

# Path to the ChromaDB persistence directory.
# This is where the vector database will be stored on disk.
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, '.chroma_db')


# --- VECTOR DB ---
# Name of the ChromaDB collection where embeddings will be stored.
COLLECTION_NAME = "local_docs"


# --- TEXT PROCESSING ---
# Maximum number of tokens for each text chunk. This is a crucial parameter
# for balancing context density and retrieval performance.
CHUNK_SIZE_TOKENS = 400


# --- OLLAMA MODELS ---
# The name of the embedding model to use from Ollama.
# This model is used to convert text chunks and user queries into vector embeddings.
# 'nomic-embed-text' is a strong, open-source choice.
EMBEDDING_MODEL = "nomic-embed-text"

# The name of the Large Language Model (LLM) to use for chat generation.
# This model generates the final response to the user based on the retrieved context.
# 'gemma:2b' is a good starting point. 'phi3:3.8b' is another great option.
LLM_MODEL = "gemma:2b"


# --- PROMPT ENGINEERING ---
# This is the master prompt template for the RAG pipeline. It is carefully
# engineered to instruct the LLM to act as a sales assistant and generate
# clean, professional, and grounded responses.
RAG_PROMPT_TEMPLATE = """
**ROLE:** You are an AI assistant for a salesperson who is on a live call with a customer.
Your goal is to generate a concise, clean, plain-text response that the salesperson can read word-for-word to the customer.

**CRITICAL INSTRUCTIONS:**
1.  **Base Your Answer on the Context:** Your entire answer MUST be derived exclusively from the [DOCUMENTATION CONTEXT] provided. Do not use any outside knowledge.
2.  **No Formatting:** DO NOT use Markdown, HTML, bullet points, or any other special formatting. The output must be a single block of plain text.
3.  **Stay in Character:** Do not break character. Do not explain your reasoning, mention the context, or refer to yourself as an AI.
4.  **Be Direct and Professional:** Phrase the response as if you are the salesperson speaking directly to the customer.
5.  **Handle Missing Information:** If the answer is not in the provided context, your entire response must be ONLY this exact phrase: "I'll need to follow up with you on that specific question."

---
[DOCUMENTATION CONTEXT]
{context}
---
[CUSTOMER'S QUESTION]
{query}
---
[YOUR RESPONSE (TO BE READ BY SALESPERSON)]
"""