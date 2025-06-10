# Sales Enablement RAG Assistant

An end-to-end demo that ingests a corpus of documentation and allows a salesperson to get quick, accurate answers to customer questions during a live call. The assistant uses a local LLM and a Retrieval-Augmented Generation (RAG) pipeline to provide answers grounded in your documentation.

---

## ✨ Why This Project Exists

This proof-of-concept (POC) is designed to showcase a modern, open-source, and locally-run stack for building a sales enablement assistant. It uses Ollama for local LLMs, ChromaDB for vector search, and Gradio for the UI.

The repository includes a sample dataset (`local_docs.zip`) containing HTML files scraped from public-facing support articles on `twilio.com`. This data is provided for demonstration purposes, and the `setup.py` script will automatically unzip it for you into the `data/local_docs` directory.

Many sales teams use powerful platforms like Gong or Highspot, but this project demonstrates how a composable and transparent AI system can be built to:

-   Parse and semantically index entire folders of HTML product documentation.
-   Provide a simple chat interface for a salesperson to ask questions in natural language.
-   Use a RAG pipeline to retrieve relevant document chunks and generate a concise, grounded answer that can be read directly to a customer.
-   Be easily extended with additional content like support tickets or call transcripts.

---

## 🔧 Key Capabilities

-   **Automated Setup**: A Python script prepares the virtual environment and pulls the necessary LLM models from Ollama.
-   **HTML Ingestion**: Ingests entire folders of HTML documentation downloaded with tools like WinHTTrack.
-   **Semantic Embeddings**: Uses Ollama (e.g., `nomic-embed-text`) to create vector embeddings of document chunks.
-   **Vector Storage**: Stores embeddings and metadata in a local ChromaDB vector database.
-   **RAG-Powered Chat**: A Gradio-based chat interface that uses a RAG pipeline to generate context-aware, grounded responses.
-   **Database Inspector**: A UI tab to view the contents of the vector database and clear it if needed.

---

## 🛠️ Setup & Running the App

### Prerequisites

-   Python 3.8+
-   [Ollama](https://ollama.com/download) installed and running on your local machine.

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Run the Setup Script

This is a one-time setup that will create a virtual environment, install Python packages, check the Ollama connection, download the required LLM models, and unzip the sample documentation.

```bash
python setup.py
```

### 3. Place HTML Documentation

Ensure you have downloaded your documentation files and placed them in the `data/local_docs/` directory. The ingestion tool will recursively search this folder for all `.html` and `.htm` files.

**Note:** The `setup.py` script will automatically unzip the included sample dataset (`local_docs.zip`) into this directory. You can replace these files with your own.

### 4. Run the Application

First, activate the virtual environment created by the setup script.

-   **On Windows:**
    ```bash
    .\.venv\Scripts\activate
    ```
-   **On macOS/Linux:**
    ```bash
    source .venv/bin/activate
    ```

Then, launch the Gradio app:

```bash
python frontend/app.py
```

The application will be available at `http://127.0.0.1:7860`.

### 5. Create Your Embeddings (First Launch)

On your first launch, you need to process the sample documentation to make it searchable.

1.  Navigate to the **📁 Ingest Docs** tab in the web UI.
2.  The folder path should already be set to `data/local_docs`.
3.  Click the **Start Ingestion** button.

This will parse all the HTML files, split them into chunks, generate vector embeddings, and store them in the local database. You only need to do this once. Once the process is complete, you can start asking questions in the **💬 Ask Assistant** tab.

---

## 🗂️ Project Structure

```
.
├── backend/
│   ├── ingest_docs.py    # Logic for parsing, chunking, and embedding HTML.
│   ├── rag_pipeline.py   # Orchestrates the RAG response generation.
│   └── vector_db.py      # Handles vector search queries to ChromaDB.
├── config/
│   └── settings.py       # Central configuration for models, paths, and prompts.
├── data/
│   └── local_docs/       # Default location for your source HTML files.
├── frontend/
│   └── app.py            # Main Gradio application, defines the UI and handles events.
├── .gitignore
├── requirements.txt      # Lists Python package dependencies.
├── setup.py              # The automated setup and installation script.
└── README.md             # You are reading it.
```

---

## 🧠 RAG Logic Breakdown

The chat interface follows a simple but powerful RAG pipeline for every query:

1.  **Embed**: The user's query is converted into a vector embedding using the `nomic-embed-text` model.
2.  **Search**: This embedding is used to perform a similarity search in ChromaDB to find the most relevant document chunks.
3.  **Augment**: The retrieved document chunks (the "context") are combined with the original query into a detailed prompt, using the template defined in `config/settings.py`.
4.  **Generate**: This final prompt is sent to the LLM (e.g., `gemma:2b` or `phi3:3.8b`), which generates a concise, grounded answer for the user.

---

## 🖥️ UI Walkthrough

The application UI is organized into three tabs:

1.  **💬 Ask Assistant**: The main chat interface. You can ask questions here. The right-hand panel will show the source documents that were used to generate the answer, providing transparency into the RAG process.
2.  **🔍 Inspect DB**: A simple database administration tool. You can inspect the number of items in the database and see a sample of their content. You can also clear all embeddings from the database to start fresh.
3.  **📁 Ingest Docs**: The interface for the ingestion pipeline. You can specify the path to your documentation folder here and click "Submit" to parse, embed, and store the documents. A log will show the progress.

---

## ❓ Sample Queries

Once you have ingested the sample data, here are a few questions you can ask the assistant to see it in action:

- `How do I connect WhatsApp to Twilio?`
- `Can I schedule messages for future delivery?`
- `How do I check pricing or SMS capability by country?`
- `How do I use Twilio with a trial account versus a full-paid account?`
- `How do I respond to an incoming SMS with TwiML?`
- `How does Twilio prevent fraud or SMS pumping?`

---

## 🧩 Key Technologies

-   **Ollama**: For running local embedding and generation language models.
-   **ChromaDB**: For a lightweight, local-first vector database.
-   **Gradio**: For building the user-friendly web UI in Python.
-   **BeautifulSoup**: For parsing and cleaning the source HTML files.

---

## 🛣️ Future Extensions
- Add support for additional content types (support tickets, sales calls).
- Use embeddings to flag missing documentation.
- Add feedback buttons for response rating.
- Incorporate agent-style memory across chat sessions.

---

## 👋 Who This Is For
- AI engineers building LLM-native product assistants.
- Sales enablement teams looking for AI copilots.
- Developers looking to learn practical RAG architecture with open tools.

---

Built as a showcase for potential technical leadership interviews.
Feel free to extend or customize it for your own needs!
