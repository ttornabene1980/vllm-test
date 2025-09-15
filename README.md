
uv python pin 3.13
uv venv
uv pip install  -r pyproject.toml
uv sync
uv lock

export PYTHONPATH=$(pwd)
e
PYTHONPATH=. streamlit run  dashboard/app.py
