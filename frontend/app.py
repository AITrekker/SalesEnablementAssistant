"""
This is the main entry point for the Gradio-based frontend of the Sales Enablement RAG Assistant.

The script performs the following actions:
1.  Checks if it's running within the correct virtual environment.
2.  Adds the project root to the system path to ensure modules are found.
3.  Defines the UI components and layout for the application inside a single,
    correctly structured Gradio Blocks container with Tabs.
4.  Launches the Gradio application.
"""
import sys
import os
import gradio as gr
import re
os.environ["OLLAMA_USE_GPU"] = "1"

# --- Virtual Environment Check ---
# Ensures the app is running in the virtual environment created by setup.py
if sys.prefix == sys.base_prefix:
    print("❌ This script is not running in a virtual environment.")
    print("   Please activate the venv by running the activation script from your terminal,")
    print("   (e.g., `.\\.venv\\Scripts\\activate` on Windows or `source .venv/bin/activate` on Linux/macOS)")
    print("   and then run `python frontend/app.py` again.")
    sys.exit(1)
print("✅ Running in virtual environment.")
# --- End Venv Check ---

# --- System Path Configuration ---
# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- End System Path Configuration ---

from backend.ingest_docs import create_ingest_ui_components, handle_folder
from backend.rag_pipeline import get_rag_response_stream
from backend.vector_db import get_db_inspection_report, clear_database_collection

def get_sample_queries():
    """
    Reads the README.md file and extracts a list of sample queries.
    """
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Find the "Sample Queries" section and extract the list items
        queries_section = re.search(r"## ❓ Sample Queries\n(.*?)(?=\n##)", content, re.DOTALL | re.IGNORECASE)
        if not queries_section:
            return []
        
        # Extract queries from markdown list items
        queries = re.findall(r"-\s*`([^`]+)`", queries_section.group(1))
        return queries
    except FileNotFoundError:
        return [] # Return empty list if README is not found
    except Exception as e:
        print(f"Error reading sample queries from README: {e}")
        return []

def build_chat_interface():
    """Defines and returns the components for the chat interface."""
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Sales Assistant",
                height=550,
                show_copy_button=True,
                bubble_full_width=False
            )
            with gr.Row():
                msg_textbox = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., How do I send an SMS message using the Python helper library?",
                    show_label=False,
                    lines=2,
                    scale=8
                )
                submit_button = gr.Button("Submit", variant="primary", scale=1)
            
            with gr.Column():
                gr.Markdown("### Sample Queries")
                sample_queries = get_sample_queries()
                for query in sample_queries:
                    btn = gr.Button(query)
                    btn.click(lambda q=query: q, None, msg_textbox)

        with gr.Column(scale=1):
            sources_display = gr.Markdown(
                "### Retrieved Sources\n\n*No sources to display yet.*",
                label="Context Sources"
            )

    def handle_chat_submission(query, history):
        print(f"\n[User Query]: {query}")
        history.append((query, None))
        yield history, gr.update(value="*Thinking...*")

        stream, sources = get_rag_response_stream(query)

        if not stream:
            history[-1] = (query, sources)
            yield history, gr.update(value=sources)
            return

        history[-1] = (query, "")
        full_response = ""
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                chunk_content = chunk['message']['content']
                full_response += chunk_content
                history[-1] = (query, full_response)
                yield history, gr.update(value=sources)
        
        print(f"[LLM Response]: {full_response}")

    # Event handlers
    submit_button.click(handle_chat_submission, [msg_textbox, chatbot], [chatbot, sources_display])
    submit_button.click(lambda: "", inputs=[], outputs=[msg_textbox])
    msg_textbox.submit(handle_chat_submission, [msg_textbox, chatbot], [chatbot, sources_display])
    msg_textbox.submit(lambda: "", inputs=[], outputs=[msg_textbox])

def build_inspector_interface():
    """Defines and returns the components for the DB inspector."""
    gr.Markdown("## 🔍 Inspect & Manage Vector Database")
    db_report_output = gr.Textbox(label="Database Inspection Report", lines=15, interactive=False)
    with gr.Row():
        inspect_button = gr.Button("Inspect Database")
        clear_db_button = gr.Button("⚠️ Clear All Embeddings", variant="stop")

    inspect_button.click(fn=get_db_inspection_report, inputs=[], outputs=[db_report_output])
    clear_db_button.click(
        fn=clear_database_collection,
        inputs=[],
        outputs=[db_report_output],
        js="() => confirm('Are you sure you want to delete all embeddings from the database? This action cannot be undone.')"
    )

def build_ingest_interface():
    """Defines and returns the components for the ingestion interface."""
    gr.Markdown("## 📁 Ingest Documentation")
    gr.Markdown("Point the tool to a local folder containing your HTML documentation. It will recursively find all `.html` files, parse them, generate embeddings, and store them in the vector database.")
    
    folder_input, log_output = create_ingest_ui_components()
    ingest_button = gr.Button("Start Ingestion", variant="primary")
    
    ingest_button.click(fn=handle_folder, inputs=[folder_input], outputs=[log_output])


# --- Main App Layout ---
def main():
    """Defines the main Gradio app layout and launches it."""
    with gr.Blocks(title="Sales Enablement RAG Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Sales Enablement RAG Assistant")
        
        with gr.Tabs():
            with gr.TabItem("💬 Ask Assistant"):
                build_chat_interface()
            
            with gr.TabItem("🔍 Inspect DB"):
                build_inspector_interface()
                
            with gr.TabItem("📁 Ingest Docs"):
                build_ingest_interface()

    print("\n--- Launching Gradio App ---")
    print(f"Navigate to: http://127.0.0.1:7860")
    demo.launch()


if __name__ == "__main__":
    main()
