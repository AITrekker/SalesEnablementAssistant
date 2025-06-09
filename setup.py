import os
import subprocess
import sys
import importlib
import threading
import time
import itertools
import zipfile

# --- Configuration ---
VENV_DIR = ".venv"

# --- Helper Functions ---

def get_python_path():
    """Gets the path to the Python executable in the virtual environment."""
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        return os.path.join(VENV_DIR, "bin", "python")

def get_pip_path():
    """Gets the path to the pip executable in the virtual environment."""
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Scripts", "pip.exe")
    else:
        return os.path.join(VENV_DIR, "bin", "pip")

# --- Core Setup Steps ---

def create_virtual_env():
    """Creates a virtual environment if it doesn't already exist."""
    if not os.path.exists(VENV_DIR):
        print(f"─ Creating virtual environment in ./{VENV_DIR}...")
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True, capture_output=True)
        print("✅ Virtual environment created successfully.")
    else:
        print("─ Virtual environment already exists. Skipping creation.")

def install_requirements():
    """Installs requirements from requirements.txt into the virtual environment."""
    pip_path = get_pip_path()
    requirements_file = "requirements.txt"
    print(f"─ Installing dependencies from {requirements_file}...")
    print(f"─ This may take a few minutes...")
    # Using capture_output=True to hide the verbose pip install log unless there's an error
    result = subprocess.run(
        [pip_path, "install", "-r", requirements_file],
        check=True,
        capture_output=True,
        text=True
    )
    print("✅ Dependencies installed successfully.")

def unzip_sample_data():
    """Checks for sample data zip and unzips it."""
    zip_path = os.path.join("data", "local_docs.zip")
    extract_path = os.path.join("data", "local_docs")
    if os.path.exists(zip_path):
        print(f"─ Found sample data at {zip_path}.")
        print(f"─ Unzipping to {extract_path}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print("✅ Sample data unzipped successfully.")
        except Exception as e:
            print(f"❌ Failed to unzip sample data: {e}")
    else:
        print("─ No sample data zip found. Skipping.")

# --- Post-Installation Checks (run inside the venv) ---

def perform_post_install_checks():
    """
    This function runs *using the venv's Python*.
    It checks for a running Ollama server and pulls the required models.
    """
    print("\n--- Running Post-Installation Checks ---")
    try:
        # Dynamically import libraries now that they're installed in the venv
        import ollama
        from config.settings import EMBEDDING_MODEL, LLM_MODEL
    except ImportError as e:
        print(f"❌ Critical Error: Could not import a required library ({e}).")
        print("   Please ensure requirements.txt is valid and all dependencies were installed correctly.")
        sys.exit(1)

    # 1. Check if the Ollama server is running and accessible
    print("─ Checking for running Ollama server...")
    try:
        ollama.list()
        print("✅ Ollama server is running and accessible.")
    except Exception:
        print("\n❌ Ollama Connection Error")
        print("   Could not connect to the Ollama server.")
        print("   Please make sure the Ollama application is running on your machine.")
        print("   You can download it from https://ollama.com/download")
        sys.exit(1)

    # 2. Check if the required models are available, and pull them if they are not
    print("─ Checking for required Ollama models...")
    required_models = {EMBEDDING_MODEL, LLM_MODEL} # Use a set for efficiency
    try:
        local_models_response = ollama.list()
        # Create a set of locally available model names (e.g., {'nomic-embed-text', 'gemma:2b'})
        local_models = {
            model['name'] for model in local_models_response.get('models', []) if 'name' in model
        }

        for model_name in required_models:
            if model_name not in local_models:
                print(f"  🟡 Model '{model_name}' not found locally. Pulling from Ollama...")
                sys.stdout.flush()

                # --- Use a threaded spinner for better UX during download ---
                stop_event = threading.Event()
                pull_thread = threading.Thread(target=lambda: ollama.pull(model=model_name, stream=False), daemon=True)
                pull_thread.start()
                spinner = itertools.cycle(['|', '/', '-', '\\'])

                while pull_thread.is_alive():
                    sys.stdout.write(f"\r  Downloading... {next(spinner)}")
                    sys.stdout.flush()
                    pull_thread.join(timeout=0.1)

                sys.stdout.write('\r' + ' ' * 30 + '\r') # Clear the spinner line
                print(f"  ✅ Successfully pulled '{model_name}'.")
            else:
                print(f"  ✅ Model '{model_name}' is already available locally.")
    except Exception as e:
        print(f"\n❌ An error occurred while checking or pulling models: {e}")
        print("   Please check your internet connection and ensure Ollama is running correctly.")
        sys.exit(1)


def main():
    """
    Main function to orchestrate the setup process.
    This script cleverly re-runs itself using the virtual environment's Python
    to perform post-install checks with the newly installed dependencies.
    """
    print("--- 🚀 Launching Sales Enablement RAG Assistant Setup 🚀 ---")
    
    # These steps run using the system's Python
    create_virtual_env()
    install_requirements()
    unzip_sample_data()

    print("\n─ Handing off to virtual environment for final checks...")
    python_venv_path = get_python_path()

    # Re-run this script with the '--post-install' flag using the venv's Python
    # This allows us to import and use the packages we just installed.
    result = subprocess.run([python_venv_path, __file__, '--post-install'])

    if result.returncode != 0:
        print("\n❌ Setup failed during post-installation checks.")
        print("   Please address the issues reported above and run `python setup.py` again.")
        sys.exit(1)

    print("\n--- 🎉 Setup Complete! 🎉 ---")
    print("You're all set. To run the application, follow these steps:")
    if sys.platform == "win32":
        print(f"1. Activate the virtual environment: .\\{VENV_DIR}\\Scripts\\activate")
        print(f"2. Run the Gradio app:              python frontend\\app.py")
    else:
        print(f"1. Activate the virtual environment: source {VENV_DIR}/bin/activate")
        print(f"2. Run the Gradio app:              python frontend/app.py")


if __name__ == "__main__":
    # This check allows the script to have two modes:
    # 1. Default mode: Run by the user, orchestrates the main setup.
    # 2. --post-install mode: Run by the main() function using the venv's Python,
    #    which has the required libraries installed.
    if '--post-install' in sys.argv:
        perform_post_install_checks()
    else:
        main() 