# Colab and Local Usage

## Google Colab with Drive

Mount Drive, change into the repository root, install `requirements.txt`, and set `GOOGLE_API_KEY` through Colab Secrets or the runtime environment. Do not place the key in a notebook cell or committed file. Set `PROJECT_ROOT` if the checkout is not at the code's detected root.

## Local clone

Create a virtual environment, install dependencies, export `GOOGLE_API_KEY`, and run commands from the repository root. Paths in active modules are derived from `config.py`; the notebook retains some clearly visible Colab/Drive setup cells because it is the authoritative experimental record.

